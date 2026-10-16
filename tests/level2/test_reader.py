import numpy as np
import pint  # noqa: F401  registers the .pint accessor the reader uses
import pint_xarray  # noqa: F401
import xarray as xr

from src.core.load import UDALoader
from src.core.model import DatasetInfo, Dimension, Mapping, ProfileInfo
from src.level2.reader import DatasetReader
from src.level2.transforms import DatasetInterpolationTransform, InterpolationParams


def test_read_profiles(mocker, sample_mapping, sample_dataarray):
    mocker.patch("src.core.load.UDALoader.load", return_value=sample_dataarray)
    shot = 30420

    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profiles = reader.read_profiles(shot, "magnetics")

    assert (profiles["ip"] == sample_dataarray).all()


def test_read_profile(mocker, sample_mapping, sample_dataarray):
    mocker.patch("src.core.load.UDALoader.load", return_value=sample_dataarray)
    shot = 30420
    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profiles = reader.read_profile(shot, "magnetics", "ip")
    assert (profiles == sample_dataarray).all()


def test_read_profile_with_background_subtraction(
    mocker, sample_mapping, sample_dataarray
):
    mocker.patch("src.core.load.UDALoader.load", return_value=sample_dataarray)
    mocker.patch(
        "src.level2.reader.DatasetReader._get_source",
        return_value=mocker.Mock(background_correction=mocker.Mock(tmin=0, tmax=5)),
    )
    shot = 30420
    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profile = reader.read_profile(shot, "magnetics", "ip")
    background_mean = sample_dataarray.isel(time=slice(0, 5)).mean(dim="time")
    expected = sample_dataarray - background_mean
    assert (profile == expected).all()


def test_interpolation_aligns_grids_across_shots():
    mapping = Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="summary/ip",
        datasets={},
    )
    transform = DatasetInterpolationTransform(None, mapping)
    params = InterpolationParams(start=-0.1, step=2.5e-4, method="zero")

    t_a = np.arange(-0.05, 0.10, 1.5e-3)
    t_b = np.arange(-0.06, 0.20, 1.3e-3)
    a = xr.DataArray(
        np.ones(len(t_a)), dims=["time"], coords={"time": t_a}, name="x"
    ).to_dataset()
    b = xr.DataArray(
        np.ones(len(t_b)), dims=["time"], coords={"time": t_b}, name="x"
    ).to_dataset()

    a_out = transform.interpolate_dimension(a, "time", params)
    b_out = transform.interpolate_dimension(b, "time", params)

    n = min(len(a_out.time), len(b_out.time))
    assert np.allclose(a_out.time.values[:n], b_out.time.values[:n])


def test_interpolation_zero_order_hold_keeps_float32_time_samples():
    mapping = Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="summary/ip",
        datasets={},
    )
    transform = DatasetInterpolationTransform(None, mapping)
    params = InterpolationParams(start=0.05, end=0.08, step=0.005, method="zero")

    t = np.array([0.05, 0.055, 0.06, 0.065, 0.07, 0.075, 0.08], dtype=np.float32)
    t = np.nextafter(t, np.float32(1.0))  # 1 step above: 0.0700000003 etc.
    t[0] = np.nextafter(np.float32(0.05), np.float32(0.0))  # 1 step below: 0.049999997
    ds = xr.DataArray(
        np.arange(len(t), dtype=float), dims=["time"], coords={"time": t}, name="x"
    ).to_dataset()

    out = transform.interpolate_dimension(ds, "time", params)

    assert out.x.values.tolist() == [0, 1, 2, 3, 4, 5, 6]


# Thomson-shaped fixtures: two signals, same pulses, two float32 axes.

N_PULSES = 145
PULSE_PERIOD = np.float32(4.1666e-3)
N_CHANNELS = 130


def _profile_axis():
    """start + n * step."""
    i = np.arange(1, N_PULSES + 1, dtype=np.float32)
    return (PULSE_PERIOD * i).astype(np.float32)


def _core_axis():
    """The same instants accumulated, so the rounding differs."""
    return np.cumsum(np.full(N_PULSES, PULSE_PERIOD, dtype=np.float32)).astype(
        np.float32
    )


def _thomson_mapping():
    radius = Dimension(source="AYC_R")
    return Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="summary/ip",
        datasets={
            "thomson_scattering": DatasetInfo(
                # no `time` block: Thomson is not time-interpolated
                interpolate={
                    "major_radius": InterpolationParams(
                        start=0.3, end=1.49, step=0.01, method="linear", fill="none"
                    )
                },
                profiles={
                    "t_e": ProfileInfo(
                        source="AYC_TE",
                        dimensions={"time": None, "major_radius": radius},
                        dimension_order=["major_radius", "time"],
                    ),
                    "t_e_core": ProfileInfo(
                        source="AYC_TE_CORE", dimensions={"time": None}
                    ),
                },
            )
        },
    )


def _fake_uda_load(shot, name, channels=None):
    radius = np.linspace(0.234, 1.517, N_CHANNELS, dtype=np.float32)
    if name == "AYC_TE_CORE":
        return xr.DataArray(
            np.ones(N_PULSES),
            dims=["time"],
            coords={"time": _core_axis()},
            attrs={"units": "eV"},
        )
    if name == "AYC_TE":
        return xr.DataArray(
            np.ones((N_PULSES, N_CHANNELS)),
            dims=["time", "radial_index"],
            coords={"time": _profile_axis(), "radial_index": radius},
            attrs={"units": "eV"},
        )
    if name == "AYC_R":
        return xr.DataArray(
            np.tile(radius, (N_PULSES, 1)),
            dims=["time", "radial_index"],
            coords={"time": _profile_axis(), "radial_index": radius},
            attrs={"units": "m"},
        )
    raise AssertionError(f"unexpected signal {name}")


def test_read_dataset_merges_divergent_float32_axes_onto_one(mocker):
    assert (_profile_axis() != _core_axis()).sum() > 100, "fixture must disagree"
    mocker.patch("src.core.load.UDALoader.load", side_effect=_fake_uda_load)

    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)
    ds = reader.read_dataset(30421, "thomson_scattering")

    assert ds.sizes["time"] == N_PULSES  # was 283, the union of both axes
    assert ds.sizes["major_radius"] == 120
    assert not ds.t_e.isnull().all(dim="major_radius").any()
    assert not ds.t_e_core.isnull().any()
