import importlib

import pytest
import xarray as xr

from src.core.load import MissingSourceError, SALLoader, UDALoader, ZarrLoader


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


@pytest.mark.parametrize(
    "name, expected",
    [
        ("magn/ipla", "/pulse/87737/ppf/signal/jetppf/magn/ipla"),
        ("chain1/efit/rxpl", "/pulse/87737/ppf/signal/chain1/efit/rxpl"),
        ("/magn/ipla/", "/pulse/87737/ppf/signal/jetppf/magn/ipla"),
    ],
)
def test_sal_signal_path(name, expected):
    assert SALLoader._signal_path(87737, name) == expected


def test_sal_signal_path_rejects_bad_source():
    with pytest.raises(MissingSourceError):
        SALLoader._signal_path(87737, "ipla")


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