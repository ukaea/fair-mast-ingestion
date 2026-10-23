import numpy as np
import pytest
import xarray as xr

from src.core.model import Mapping
from src.level2.transforms import (
    BackgroundSubtractionTransform,
    DatasetInterpolationTransform,
    FFTDecomposeTransform,
    InterpolationParams,
    transform_registry,
)


def _mapping(**kwargs):
    defaults = dict(
        facility="MAST", default_loader="uda", plasma_current="summary/ip", datasets={}
    )
    defaults.update(kwargs)
    return Mapping(**defaults)


def test_interpolate_dimension_raises_without_start_or_default(mocker):
    transform = DatasetInterpolationTransform(
        mocker.Mock(), _mapping(default_start=None)
    )
    params = InterpolationParams(step=2.5e-4, method="linear")

    t = np.arange(0, 1, 0.1)
    ds = xr.DataArray(
        np.ones(len(t)), dims=["time"], coords={"time": t}, name="x"
    ).to_dataset()

    with pytest.raises(ValueError, match="default_start"):
        transform.interpolate_dimension(ds, "time", params)


def test_interpolate_dimension_raises_without_step(mocker):
    transform = DatasetInterpolationTransform(mocker.Mock(), _mapping())
    params = InterpolationParams(start=0.0, method="linear")

    t = np.arange(0, 1, 0.1)
    ds = xr.DataArray(
        np.ones(len(t)), dims=["time"], coords={"time": t}, name="x"
    ).to_dataset()

    with pytest.raises(ValueError, match="step"):
        transform.interpolate_dimension(ds, "time", params)


def test_interpolate_dimension_uses_mapping_default_start(mocker):
    transform = DatasetInterpolationTransform(
        mocker.Mock(), _mapping(default_start=-0.5)
    )
    params = InterpolationParams(step=0.5, end=0.5, method="linear")

    t = np.arange(-1, 1, 0.1)
    ds = xr.DataArray(
        np.ones(len(t)), dims=["time"], coords={"time": t}, name="x"
    ).to_dataset()

    out = transform.interpolate_dimension(ds, "time", params)
    np.testing.assert_allclose(out.time.values, [-0.5, 0.0, 0.5])


def test_interpolate_dimension_method_none_returns_unmodified(mocker):
    transform = DatasetInterpolationTransform(mocker.Mock(), _mapping())
    params = InterpolationParams(start=0.0, step=0.5, method="none")

    t = np.arange(0, 1, 0.1)
    ds = xr.DataArray(
        np.ones(len(t)), dims=["time"], coords={"time": t}, name="x"
    ).to_dataset()

    out = transform.interpolate_dimension(ds, "time", params)
    assert out.equals(ds)


def test_interpolate_dimension_ffill(mocker):
    transform = DatasetInterpolationTransform(mocker.Mock(), _mapping())
    params = InterpolationParams(
        start=0.0, end=3.0, step=1.0, method="none", fill="ffill"
    )

    t = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, np.nan, np.nan, 4.0])
    ds = xr.DataArray(values, dims=["time"], coords={"time": t}, name="x").to_dataset()

    out = transform.interpolate_dimension(ds, "time", params)
    np.testing.assert_allclose(out["x"].values, [1.0, 1.0, 1.0, 4.0])


def test_interpolate_dimension_dropna(mocker):
    transform = DatasetInterpolationTransform(mocker.Mock(), _mapping())
    params = InterpolationParams(
        start=0.0, end=3.0, step=1.0, method="none", dropna=True
    )

    t = np.array([0.0, 1.0, 2.0, 3.0])
    values = np.array([1.0, np.nan, np.nan, 4.0])
    ds = xr.DataArray(values, dims=["time"], coords={"time": t}, name="x").to_dataset()

    out = transform.interpolate_dimension(ds, "time", params)
    assert len(out.time) == 2


def test_interpolate_dimensions_skips_dims_without_params(mocker):
    transform = DatasetInterpolationTransform(mocker.Mock(), _mapping())
    ds = xr.Dataset()
    out = transform.interpolate_dimensions(
        ds, dimensions={"time": None}, interpolate_params=None
    )
    assert out is ds


def test_fft_decompose_transform():
    times = np.linspace(0, 1, 512)
    data = np.sin(2 * np.pi * 50 * times)
    signal = xr.DataArray(data, dims=["time"], coords={"time": times}, name="probe")

    transform = FFTDecomposeTransform(nperseg=128)
    result = transform.transform_array("probe", signal)

    assert isinstance(result, xr.Dataset)
    assert "probe_spectrogram" in result.data_vars
    assert "probe_angles" in result.data_vars
    assert result["probe_spectrogram"].dims == ("frequency", "time")


def test_background_subtraction_no_time_dim_returns_unchanged():
    data = xr.DataArray(np.arange(5.0), dims=["channel"], name="x")
    transform = BackgroundSubtractionTransform(start=0, end=2)

    result = transform.transform_array(data)
    assert (result == data).all()


def test_background_subtraction_subtracts_mean_of_window():
    time = np.arange(10.0)
    data = xr.DataArray(time + 10, dims=["time"], coords={"time": time}, name="x")
    transform = BackgroundSubtractionTransform(start=0, end=2)

    result = transform.transform_array(data)
    background = data.isel(time=slice(0, 2)).mean(dim="time")
    assert (result == data - background).all()


def test_transform_registry_has_fftdecompose():
    transform = transform_registry.create("fftdecompose", nperseg=64)
    assert isinstance(transform, FFTDecomposeTransform)
    assert transform.nperseg == 64
