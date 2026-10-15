"""Thomson scattering checks against the MAST-U level 2 mapping.

The UDA client is mocked, so these tests run without a UDA server. The fake
signals copy the layout of MAST-U shot 52625:

- ayc (core): 130 polychromators (IDs 301-430), 150 Hz, first laser at -26.6 ms.
- ayd (divertor): 12 polychromators (IDs 201-212) with radius decreasing with
  channel index, 30 Hz, first laser at 0.19 ms.

Each laser sample sits slightly after a grid point, which is the case that
makes a non-causal or misaligned time hold visible.
"""

from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pint_xarray  # noqa: F401 - registers the .pint accessor that the reader uses
import pytest

from src.core.load import UDALoader
from src.core.model import load_model
from src.level2.reader import DatasetReader

MAPPING_FILE = Path(__file__).parents[2] / "mappings" / "level2" / "mastu.yml"
SHOT = 52625
TMAX = 1.3  # stands in for the plasma current end time

CORE_RATE = 150.0
CORE_T0 = -0.0265663
CORE_N_TIME = 183
CORE_IDS = np.arange(301, 431)
CORE_R = np.linspace(0.3064, 1.4805, CORE_IDS.size)

DIV_RATE = 30.0
DIV_T0 = 0.0001902
DIV_N_TIME = 36
DIV_IDS = np.arange(201, 213)
DIV_R = np.array(
    [
        1.4389,
        1.3963,
        1.3474,
        1.3008,
        1.2567,
        1.2150,
        1.1760,
        1.1398,
        1.1067,
        1.0772,
        1.0516,
        1.0304,
    ]
)


def _dim(label, data, units):
    return SimpleNamespace(label=label, data=np.asarray(data), units=units)


def _signal(data, dims, units):
    data = np.asarray(data, dtype=float)
    return SimpleNamespace(
        data=data,
        errors=np.zeros_like(data),
        dims=dims,
        units=units,
        description="fake signal",
        label="fake",
    )


def _sample_ids(n_time, n_channel):
    """Unique value per (time, channel), so a value identifies its raw sample."""
    return (np.arange(n_time)[:, None] + 1) * 1000.0 + np.arange(n_channel)[None, :]


def _profile_signals(source, time, ids, radius):
    time_dim = _dim("Time", time, "s")
    channel_dim = _dim("polychromator", ids, "")
    dims = [time_dim, channel_dim]
    values = _sample_ids(time.size, ids.size)
    return {
        f"{source}/time": _signal(time, [time_dim], "s"),
        f"{source}/r": _signal(np.tile(radius, (time.size, 1)), dims, "m"),
        f"{source}/t_e": _signal(values, dims, "eV"),
        f"{source}/n_e": _signal(values * 1e17, dims, "m^-3"),
        f"{source}/p_e": _signal(values, dims, "Pa"),
    }


class FakeUDAClient:
    def __init__(self):
        core_time = CORE_T0 + np.arange(CORE_N_TIME) / CORE_RATE
        div_time = DIV_T0 + np.arange(DIV_N_TIME) / DIV_RATE
        time_dim = [_dim("Time", core_time, "s")]

        self.signals = {
            **_profile_signals("ayc", core_time, CORE_IDS, CORE_R),
            **_profile_signals("ayd", div_time, DIV_IDS, DIV_R),
            "ayc/t_e_core": _signal(np.arange(CORE_N_TIME) + 1.0, time_dim, "eV"),
            "ayc/n_e_core": _signal(
                (np.arange(CORE_N_TIME) + 1.0) * 1e19, time_dim, "m^-3"
            ),
        }
        self.core_time = core_time
        self.div_time = div_time

    def get(self, name, shot):
        import pyuda

        key = name.strip("/").lower()
        if key not in self.signals:
            raise pyuda.ServerException(
                "The requested dataset or attribute does not exist: check name and case"
            )
        return self.signals[key]


@pytest.fixture
def fake_client(mocker):
    client = FakeUDAClient()
    mocker.patch.object(UDALoader, "_get_client", return_value=client)
    return client


@pytest.fixture
def reader(fake_client):
    mapping = load_model(MAPPING_FILE)
    mapping.tmax = TMAX
    return DatasetReader(mapping, UDALoader(), skip_geometry=True)


def _last_sample_at_or_before(raw_time, out_time):
    return np.searchsorted(raw_time, out_time, side="right") - 1


@pytest.mark.parametrize(
    "dataset_name, radius",
    [("thomson_scattering", CORE_R), ("thomson_scattering_divertor", DIV_R)],
)
def test_profile_major_radius_is_position_not_channel_id(reader, dataset_name, radius):
    profile = reader.read_profile(SHOT, dataset_name, "t_e")

    assert np.allclose(profile.major_radius.values, np.sort(radius))
    assert profile.major_radius.attrs["units"] == "m"


def test_core_profiles_have_data_after_radial_regrid(reader):
    dataset = reader.read_dataset(SHOT, "thomson_scattering")

    for name in ["t_e", "n_e", "p_e"]:
        has_data = dataset[name].notnull().any("time")
        assert has_data.any(), f"{name} is all NaN after regridding"

        radius_with_data = dataset.major_radius.values[has_data.values]
        assert radius_with_data.min() >= CORE_R.min()
        assert radius_with_data.max() <= CORE_R.max()


def test_divertor_keeps_native_channel_radii(reader):
    dataset = reader.read_dataset(SHOT, "thomson_scattering_divertor")

    assert dataset.sizes["major_radius"] == DIV_IDS.size
    assert np.allclose(dataset.major_radius.values, np.sort(DIV_R))


@pytest.mark.parametrize(
    "dataset_name, rate",
    [("thomson_scattering", CORE_RATE), ("thomson_scattering_divertor", DIV_RATE)],
)
def test_time_step_matches_laser_rate(reader, dataset_name, rate):
    dataset = reader.read_dataset(SHOT, dataset_name)

    assert np.allclose(np.diff(dataset.time.values), 1 / rate)


@pytest.mark.parametrize(
    "dataset_name, variable",
    [("thomson_scattering", "t_e_core"), ("thomson_scattering_divertor", "t_e")],
)
def test_time_hold_is_causal(reader, fake_client, dataset_name, variable):
    dataset = reader.read_dataset(SHOT, dataset_name)
    if dataset_name == "thomson_scattering":
        raw_time = fake_client.core_time
        raw = fake_client.signals["ayc/t_e_core"].data
    else:
        raw_time = fake_client.div_time
        order = np.argsort(DIV_R)
        raw = fake_client.signals["ayd/t_e"].data[:, order]

    out_time = dataset.time.values
    out = dataset[variable].transpose("time", ...).values
    index = _last_sample_at_or_before(raw_time, out_time)
    held = (index >= 0) & (out_time <= raw_time[-1])

    # Each output holds the last sample taken at or before its own time
    assert np.array_equal(out[held], raw[index[held]])
    # No sample is reported before it was taken
    assert np.isnan(out[index < 0]).all()
    # Each sample is used at most once, so no sample is duplicated
    assert np.unique(index[held]).size == held.sum()
