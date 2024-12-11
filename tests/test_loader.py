import xarray as xr

from src.load import UDALoader


def test_load_uda_signal():
    loader = UDALoader()
    signal = loader.load(30420, "AMC_PLASMA CURRENT")
    assert isinstance(signal, xr.Dataset)
    assert "AMC_PLASMA CURRENT" in signal
    assert "AMC_PLASMA CURRENT_error" in signal


def test_load_uda_image():
    loader = UDALoader()
    signal = loader.load(30420, "RBB")
    assert isinstance(signal, xr.Dataset)
    assert "RBB" in signal
