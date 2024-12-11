import pytest
import importlib
import xarray as xr

from src.load import UDALoader

uda_available = not importlib.util.find_spec("pyuda")


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_load_uda_signal():
    loader = UDALoader()
    signal = loader.load(30420, "AMC_PLASMA CURRENT")
    assert isinstance(signal, xr.Dataset)
    assert "AMC_PLASMA CURRENT" in signal
    assert "AMC_PLASMA CURRENT_error" in signal


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_load_uda_image():
    loader = UDALoader()
    signal = loader.load(30420, "RBB")
    assert isinstance(signal, xr.Dataset)
    assert "RBB" in signal
