import importlib

import numpy as np
import pytest
import xarray as xr

from src.core.load import MissingSourceError, SALLoader, UDALoader, ZarrLoader
from src.core.model import Channel, Source


def try_uda():
    try:
        import pyuda

        client = pyuda.Client()
        client.get("ip", 30421)
        return False
    except Exception:
        return True


@pytest.mark.skipif(try_uda(), reason="Pyuda client unavailable")
def test_load_uda():
    loader = UDALoader()
    signal = loader.load(30420, "ip")
    assert isinstance(signal, xr.DataArray)


@pytest.mark.skipif(
    not importlib.util.find_spec("jet"), reason="requires the Jet SAL library"
)
def test_load_sal():
    loader = SALLoader()
    signal = loader.load(87737, "magn/ipla")
    assert isinstance(signal, xr.DataArray)
    assert signal.attrs["units"] == "Amps"
    assert signal.dim_0.attrs["units"] == "Secs"


def test_load_zarr_remote():
    config = {
        "base_path": "s3://mast/level1/shots",
        "protocol": "simplecache",
        "target_protocol": "s3",
        "target_options": {"anon": True, "endpoint_url": "https://s3.echo.stfc.ac.uk"},
    }

    loader = ZarrLoader(**config)
    signal = loader.load(30420, "amc/plasma_current")
    assert isinstance(signal, xr.DataArray)

@pytest.mark.parametrize(
    "channels, expected_values, expected_template",
    [
        (["xmc/CC/MT/201", "xmc/CC/MT/206"], ["201", "206"], "xmc/CC/MT/{channel}"),
        (["AMB_FL/CC03", "AMB_FL/P3L/4"], ["CC03", "P3L/4"], "AMB_FL/{channel}"),
        (["AMC_P2IL FEED CURRENT", "AMC_P3L FEED CURRENT"], ["P2IL", "P3L"], "AMC_{channel} FEED CURRENT"),
        (["ALPHA/1", "BETA/2"], ["ALPHA/1", "BETA/2"], None),
    ],
)
def test_extract_channel_template(channels, expected_values, expected_template):
    values, template = UDALoader._extract_channel_template(channels)
    assert values == expected_values
    assert template == expected_template


def test_mask_channel(mocker, sample_dataarray):
    """A masked channel keeps its place on the axis but holds no data."""
    mocker.patch("src.core.load.UDALoader.load_signal", return_value=sample_dataarray)
    channels = [
        Channel(name="AMB_FL/CC03"),
        Channel(name="AMB_FL/CC04", scale=-1, mask=True),
        Channel(name="AMB_FL/CC05", scale=3),
    ]

    signal = UDALoader().load_channels(30420, "AMB_FL", channels)

    assert list(signal.coords["channels"].values) == ["CC03", "CC04", "CC05"]
    assert np.isnan(signal.isel(channels=1)).all()
    assert (signal.isel(channels=0) == sample_dataarray).all()
    assert (signal.isel(channels=2) == sample_dataarray * 3).all()


def test_missing_first_channel_keeps_units(mocker):
    """A missing first channel must not drop the units (AMM_P2LARM1 on 13105)."""

    def load_signal(shot, name):
        if name == "AMM_P2LARM1":
            raise MissingSourceError(name)
        return xr.DataArray(
            np.ones(3), dims=["time"], coords={"time": [0.0, 1.0, 2.0]}, attrs={"units": "kA"}
        )

    mocker.patch("src.core.load.UDALoader.load_signal", side_effect=load_signal)
    channels = [Channel(name="AMM_P2LARM1"), Channel(name="AMM_P2LARM2")]

    signal = UDALoader().load_channels(13105, "AMM_P2LARM", channels)

    assert signal.attrs["units"] == "kA"
    assert signal.isel(channels=0).isnull().all()


@pytest.mark.parametrize(
    "channels, expected_masks",
    [
        (["AMB_FL/CC03", "AMB_FL/CC04"], [False, False]),
        (["AMB_FL/CC03", {"AMB_FL/CC04": {"mask": True}}], [False, True]),
        ([{"AMB_FL/CC04": {"scale": -1}}], [False]),
    ],
)
def test_channel_mask_parsing(channels, expected_masks):
    source = Source(name="AMB_FL", channels=channels)
    assert [c.mask for c in source.channels] == expected_masks