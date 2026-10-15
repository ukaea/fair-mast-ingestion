import numpy as np
import xarray as xr

from src.core.load import UDALoader
from src.core.model import DatasetInfo, Mapping, ProfileInfo
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


def test_read_profile_with_background_subtraction(mocker, sample_mapping, sample_dataarray):
    mocker.patch("src.core.load.UDALoader.load", return_value=sample_dataarray)
    mocker.patch("src.level2.reader.DatasetReader._get_source", return_value=mocker.Mock(
        background_correction=mocker.Mock(tmin=0, tmax=5)
    ))
    shot = 30420
    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profile = reader.read_profile(shot, "magnetics", "ip")
    background_mean = sample_dataarray.isel(time=slice(0, 5)).mean(dim="time")
    expected = sample_dataarray - background_mean
    assert (profile == expected).all()

def test_interpolation_aligns_grids_across_shots():
    mapping = Mapping(
        facility="MAST", default_loader="uda",
        plasma_current="summary/ip", datasets={},
    )
    transform = DatasetInterpolationTransform(None, mapping)
    params = InterpolationParams(start=-0.1, step=2.5e-4, method="zero")

    t_a = np.arange(-0.05, 0.10, 1.5e-3)
    t_b = np.arange(-0.06, 0.20, 1.3e-3)
    a = xr.DataArray(np.ones(len(t_a)), dims=["time"], coords={"time": t_a}, name="x").to_dataset()
    b = xr.DataArray(np.ones(len(t_b)), dims=["time"], coords={"time": t_b}, name="x").to_dataset()

    a_out = transform.interpolate_dimension(a, "time", params)
    b_out = transform.interpolate_dimension(b, "time", params)

    n = min(len(a_out.time), len(b_out.time))
    assert np.allclose(a_out.time.values[:n], b_out.time.values[:n])


def _hold(values, times, params):
    mapping = Mapping(
        facility="MAST", default_loader="uda",
        plasma_current="summary/ip", datasets={},
    )
    transform = DatasetInterpolationTransform(None, mapping)
    data = xr.DataArray(values, dims=["time"], coords={"time": times}, name="x")
    return transform.interpolate_dimension(data, "time", params)


def test_zero_hold_keeps_samples_on_grid_points_with_rounding_noise():
    step = 1e-4
    times = -0.1 + np.arange(1000) * step
    # Rounding puts some samples just after their grid point
    times = times + np.where(np.arange(1000) % 2 == 0, 1e-9, -1e-9)
    params = InterpolationParams(start=-0.1, end=-0.1 + 999 * step, step=step, method="zero")

    out = _hold(np.arange(1000.0), times, params)

    assert np.array_equal(out.values, np.arange(1000.0))


def test_zero_hold_keeps_last_sample_slightly_before_its_grid_point():
    step = 1e-2
    times = -0.1 + np.arange(10) * step
    times[-1] -= 1e-9  # the last sample sits just before its grid point
    params = InterpolationParams(start=-0.1, end=-0.1 + 9 * step, step=step, method="zero")

    out = _hold(np.arange(10.0), times, params)

    assert out.values[-1] == 9.0
    assert np.unique(out.values).size == 10


def test_native_step_comes_from_the_data():
    mapping = Mapping(
        facility="MAST", default_loader="uda", plasma_current="summary/ip",
        datasets={}, tmax=0.5,
    )
    dataset_params = DatasetInfo(
        profiles={
            "a": ProfileInfo(source="a", dimensions={"time": None}),
            "b": ProfileInfo(source="b", dimensions={"time": None}),
        },
        interpolate={"time": InterpolationParams(start=-0.1, step="native", method="zero")},
    )
    t_a = -0.09 + np.arange(60) * 0.01
    t_b = -0.05 + np.arange(110) * 0.00626
    profiles = {
        "a": xr.DataArray(np.arange(60.0), dims=["time"], coords={"time": t_a}, name="a"),
        "b": xr.DataArray(np.arange(110.0), dims=["time"], coords={"time": t_b}, name="b"),
    }

    out = DatasetInterpolationTransform(dataset_params, mapping).transform(profiles)

    # One shared grid, at the faster of the two native steps
    assert np.allclose(np.diff(out.time.values), 0.00626)


def _native_time_transform():
    mapping = Mapping(
        facility="MAST", default_loader="uda", plasma_current="summary/ip",
        datasets={},
    )
    dataset_params = DatasetInfo(
        profiles={
            "t_e": ProfileInfo(source="t_e", dimensions={"time": None}),
            "n_e": ProfileInfo(source="n_e", dimensions={"time": None}),
        },
        interpolate={"time": InterpolationParams(method="none")},
    )
    return DatasetInterpolationTransform(dataset_params, mapping)


def _profile(name, times):
    return xr.DataArray(
        np.arange(times.size, dtype=float), dims=["time"],
        coords={"time": times}, name=name,
    )


def test_native_time_aligns_floating_point_differences():
    # Jittered laser times, read with rounding noise for one of the profiles
    times = 0.1 + np.arange(50) * 6.67e-3
    times += np.random.default_rng(0).uniform(-1e-3, 1e-3, times.size)
    noisy = times + np.random.default_rng(1).uniform(-1e-12, 1e-12, times.size)
    profiles = {"t_e": _profile("t_e", times), "n_e": _profile("n_e", noisy)}

    out = _native_time_transform().transform(profiles)

    assert np.array_equal(out.time.values, times)
    assert out.t_e.notnull().all() and out.n_e.notnull().all()
    assert np.array_equal(out.n_e.values, np.arange(times.size))


def test_native_time_keeps_samples_that_are_not_shared():
    times = 0.1 + np.arange(50) * 6.67e-3
    extra = np.append(times, times[-1] + 6.67e-3)
    profiles = {"t_e": _profile("t_e", times), "n_e": _profile("n_e", extra)}

    out = _native_time_transform().transform(profiles)

    assert np.array_equal(out.time.values, extra)
    assert out.t_e.isel(time=-1).isnull() and out.n_e.notnull().all()