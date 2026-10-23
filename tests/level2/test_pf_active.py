import numpy as np
import pint_xarray  # noqa: F401
import pytest
import xarray as xr

from src.core.load import UDALoader
from src.core.model import load_model
from src.level2.reader import DatasetReader

SOLENOID = "AMC_SOL CURRENT"

EXPECTED_TURNS = {
    "AMC_P2IL FEED CURRENT": 12,
    "AMC_P2IU FEED CURRENT": 12,
    "AMC_P2OL FEED CURRENT": 8,
    "AMC_P2OU FEED CURRENT": 8,
    "AMC_P3L FEED CURRENT": 8,
    "AMC_P3U FEED CURRENT": 8,
    "AMC_P4L FEED CURRENT": 23,
    "AMC_P4U FEED CURRENT": 23,
    "AMC_P5L FEED CURRENT": 23,
    "AMC_P5U FEED CURRENT": 23,
    "AMC_P6L CURRENT": 4,
    "AMC_P6U CURRENT": 4,
}


@pytest.fixture(scope="module")
def mast_mapping():
    return load_model("mappings/level2/mast.yml")


@pytest.fixture
def coil_current_source(mast_mapping):
    profile = mast_mapping.datasets["pf_active"].profiles["coil_current"]
    return profile, profile.source[0]


def test_coil_current_units_are_per_turn(coil_current_source):
    profile, _ = coil_current_source
    assert profile.units == "kA/turn"
    assert profile.target_units is None


def test_coil_current_only_solenoid_is_scaled(coil_current_source):
    _, source = coil_current_source
    scales = {c.name: c.scale for c in source.channels}
    assert scales.pop(SOLENOID) == 0.5
    assert all(scale == 1.0 for scale in scales.values())


def test_coil_current_turns_attribute(coil_current_source):
    _, source = coil_current_source
    channel_names = {c.name for c in source.channels}
    assert source.attributes["turns"] == EXPECTED_TURNS
    assert set(EXPECTED_TURNS) <= channel_names


def _fake_signal(shot_num, name):
    # UDA gives the P6 units as "kA.turn" and all other channels as "kA".
    units = "kA.turn" if name.startswith("AMC_P6") else "kA"
    time = np.linspace(-0.1, 0.5, 50)
    return xr.DataArray(
        np.full(len(time), 10.0),
        dims=["time"],
        coords={"time": time},
        attrs={"units": units},
    )


def test_read_coil_current(mocker, mast_mapping):
    mocker.patch.object(UDALoader, "load_signal", side_effect=_fake_signal)
    reader = DatasetReader(mast_mapping, UDALoader(), skip_geometry=True)

    item = reader.read_profile(30420, "pf_active", "coil_current")

    assert item.attrs["units"] == "kA / turn"
    assert item.attrs["turns"] == EXPECTED_TURNS

    values = item.max(dim="time").values
    channel_names = [
        c.name
        for c in mast_mapping.datasets["pf_active"]
        .profiles["coil_current"]
        .source[0]
        .channels
    ]
    expected = [5.0 if name == SOLENOID else 10.0 for name in channel_names]
    np.testing.assert_allclose(values, expected)
