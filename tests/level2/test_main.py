import numpy as np
import pytest
import xarray as xr
import yaml
import zarr

from src.core.config import ReaderConfig
from src.core.load import UDALoader
from src.core.model import Mapping
from src.level2.main import (
    check_plasma_current,
    get_default_loader,
    load_config_file,
    load_mapping_file,
    process_shot,
    running_mean,
    safe_process_shot,
    set_mapping_time_bounds,
    trim_ip_range,
)


def test_running_mean_matches_naive_average():
    x = np.arange(10, dtype=float)
    result = running_mean(x, N=3)
    expected = np.array([np.mean(x[i : i + 3]) for i in range(len(x) - 2)])
    np.testing.assert_allclose(result, expected)


@pytest.fixture
def trapezoid_ip_dataarray():
    """A plasma-current-shaped signal: near-zero, ramps up, flat-tops well
    above the 200kA detection threshold used by check_plasma_current/
    trim_ip_range, then ramps back down to near-zero. The zero/ramp
    sections are longer than trim_ip_range's smoothing window (NN_adapt =
    300 * dt / 2e-4 = 375 samples here) so the running mean actually drops
    below threshold at the edges. dt matches the pipeline's default.
    """
    dt = 2.5e-4
    zero_len, ramp_len, flat_len = 600, 600, 800
    values = np.concatenate(
        [
            np.zeros(zero_len),
            np.linspace(0, 300_000, ramp_len),
            np.full(flat_len, 300_000.0),
            np.linspace(300_000, 0, ramp_len),
            np.zeros(zero_len),
        ]
    ).astype(np.float32)
    n = len(values)
    time = np.arange(n) * dt - (n * dt) / 2
    return xr.DataArray(values, dims=["time"], coords={"time": time}, name="ip")


def test_check_plasma_current_detects_flattop(trapezoid_ip_dataarray):
    assert check_plasma_current(trapezoid_ip_dataarray, tdelta=2.5e-4)


def test_check_plasma_current_false_for_no_current():
    time = np.arange(1000) * 2.5e-4
    flat_zero = xr.DataArray(np.zeros(1000), dims=["time"], coords={"time": time})
    assert not check_plasma_current(flat_zero, tdelta=2.5e-4)


def test_trim_ip_range_removes_leading_and_trailing_zero_current(
    trapezoid_ip_dataarray,
):
    trimmed = trim_ip_range(trapezoid_ip_dataarray, delta_time=2.5e-4)
    # The leading/trailing all-zero regions should have been trimmed away.
    assert len(trimmed) < len(trapezoid_ip_dataarray)
    # What remains should still contain the flat-top current.
    assert np.abs(trimmed.values).max() > 250_000


def test_load_mapping_file_missing_exits():
    with pytest.raises(SystemExit):
        load_mapping_file("does/not/exist.yml")


def test_load_mapping_file_success():
    mapping = load_mapping_file("mappings/level2/mast.yml")
    assert isinstance(mapping, Mapping)


def test_load_config_file_missing_exits():
    with pytest.raises(SystemExit):
        load_config_file("does/not/exist.yml")


def test_load_config_file_success():
    config = load_config_file("configs/level2.yml")
    assert "uda" in config.readers


def test_get_default_loader_builds_uda_loader():
    loader = get_default_loader(ReaderConfig(type="uda"))
    assert isinstance(loader, UDALoader)


def test_set_mapping_time_bounds_sets_tmax(mocker, trapezoid_ip_dataarray):
    mapping = Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="magnetics/ip",
        datasets={},
    )
    mocker.patch(
        "src.level2.main.DatasetReader.read_profile",
        return_value=trapezoid_ip_dataarray,
    )

    set_mapping_time_bounds(mapping, shot=30420, tdelta=2.5e-4, loader=mocker.Mock())

    assert mapping.tmax is not None
    assert mapping.tmax < trapezoid_ip_dataarray.time.values.max()


def test_set_mapping_time_bounds_raises_when_no_plasma_current(mocker):
    mapping = Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="magnetics/ip",
        datasets={},
    )
    time = np.arange(1000) * 2.5e-4
    zero_current = xr.DataArray(np.zeros(1000), dims=["time"], coords={"time": time})
    mocker.patch(
        "src.level2.main.DatasetReader.read_profile", return_value=zero_current
    )

    with pytest.raises(RuntimeError, match="No Plasma Current"):
        set_mapping_time_bounds(
            mapping, shot=30420, tdelta=2.5e-4, loader=mocker.Mock()
        )


def test_set_mapping_time_bounds_force_ip_check_suppresses_error(mocker):
    mapping = Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="magnetics/ip",
        datasets={},
    )
    time = np.arange(1000) * 2.5e-4
    zero_current = xr.DataArray(np.zeros(1000), dims=["time"], coords={"time": time})
    mocker.patch(
        "src.level2.main.DatasetReader.read_profile", return_value=zero_current
    )

    # Should not raise, just warn.
    set_mapping_time_bounds(
        mapping, shot=30420, tdelta=2.5e-4, loader=mocker.Mock(), force=True
    )


def _write_mapping_yaml(path, dataset_names):
    datasets = {
        name: {
            "profiles": {
                "signal": {
                    "source": "signal_name",
                    "description": f"{name} signal",
                    "dimensions": {"time": None},
                }
            }
        }
        for name in dataset_names
    }
    content = {
        "facility": "MAST",
        "default_loader": "uda",
        "plasma_current": f"{dataset_names[0]}/signal",
        "datasets": datasets,
    }
    path.write_text(yaml.safe_dump(content))


def _write_config_yaml(path, output_path):
    content = {
        "readers": {"uda": {"type": "uda"}},
        "writer": {
            "type": "zarr",
            "options": {"output_path": str(output_path)},
        },
    }
    path.write_text(yaml.safe_dump(content))


@pytest.fixture
def process_shot_kwargs(tmp_path, trapezoid_ip_dataarray):
    mapping_file = tmp_path / "mapping.yml"
    config_file = tmp_path / "config.yml"
    output_path = tmp_path / "output"

    _write_mapping_yaml(mapping_file, ["magnetics", "diagnostics"])
    _write_config_yaml(config_file, output_path)

    return dict(
        mapping_file=str(mapping_file),
        config_file=str(config_file),
        output_path=None,
        dt=2.5e-4,
        include_datasets=[],
        exclude_datasets=[],
        verbose=False,
        force_ip_check=False,
        skip_geometry=False,
    ), output_path


def test_process_shot_writes_all_datasets(
    mocker, process_shot_kwargs, trapezoid_ip_dataarray
):
    kwargs, output_path = process_shot_kwargs
    mocker.patch("src.core.load.UDALoader.load", return_value=trapezoid_ip_dataarray)

    process_shot(30420, **kwargs)

    out_file = output_path / "30420.zarr"
    assert out_file.exists()
    group_keys = list(zarr.open_group(str(out_file), mode="r").keys())
    assert set(group_keys) == {"magnetics", "diagnostics"}


def test_process_shot_respects_exclude_datasets(
    mocker, process_shot_kwargs, trapezoid_ip_dataarray
):
    kwargs, output_path = process_shot_kwargs
    kwargs["exclude_datasets"] = ["diagnostics"]
    mocker.patch("src.core.load.UDALoader.load", return_value=trapezoid_ip_dataarray)

    process_shot(30420, **kwargs)

    out_file = output_path / "30420.zarr"
    group_keys = list(zarr.open_group(str(out_file), mode="r").keys())
    assert group_keys == ["magnetics"]


def test_process_shot_respects_include_datasets(
    mocker, process_shot_kwargs, trapezoid_ip_dataarray
):
    kwargs, output_path = process_shot_kwargs
    kwargs["include_datasets"] = ["magnetics"]
    mocker.patch("src.core.load.UDALoader.load", return_value=trapezoid_ip_dataarray)

    process_shot(30420, **kwargs)

    out_file = output_path / "30420.zarr"
    group_keys = list(zarr.open_group(str(out_file), mode="r").keys())
    assert group_keys == ["magnetics"]


def test_safe_process_shot_swallows_exceptions(mocker):
    mocker.patch("src.level2.main.process_shot", side_effect=RuntimeError("boom"))
    # Should not raise.
    safe_process_shot(30420)
