import importlib

import pytest
import xarray as xr

from src.core.load import UDALoader

uda_available = not importlib.util.find_spec("pyuda")


@pytest.mark.skip(reason="Pyuda client unavailable")
def test_load_uda_signal():
    loader = UDALoader(include_error=True)
    signal = loader.load(30420, "AMC_PLASMA CURRENT")
    assert isinstance(signal, xr.Dataset)
    assert "AMC_PLASMA CURRENT" in signal
    assert "AMC_PLASMA CURRENT_error" in signal


@pytest.mark.skip(reason="Pyuda client unavailable")
def test_load_uda_image():
    loader = UDALoader()
    signal = loader.load(30420, "RBB")
    assert isinstance(signal, xr.Dataset)
    assert "RBB" in signal
