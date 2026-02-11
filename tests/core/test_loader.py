import importlib

import pytest
import xarray as xr

from src.core.load import SALLoader, UDALoader, ZarrLoader


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
