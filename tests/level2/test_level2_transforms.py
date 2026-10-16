import numpy as np
import xarray as xr

from src.core.model import DatasetInfo, InterpolationParams, Mapping, ProfileInfo
from src.level2.transforms import DatasetInterpolationTransform


def _mapping() -> Mapping:
    return Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="summary/ip",
        datasets={},
    )


def _series(times: np.ndarray) -> xr.Dataset:
    """Values equal their own index, so a stale hold is obvious."""
    return xr.DataArray(
        np.arange(len(times), dtype=float),
        dims=["time"],
        coords={"time": times},
        name="x",
    ).to_dataset()


def _merge(profiles: dict) -> xr.Dataset:
    """The real merge path, with no interpolation configured."""
    params = DatasetInfo(
        profiles={
            name: ProfileInfo(source=name, dimensions={"time": None})
            for name in profiles
        }
    )
    return DatasetInterpolationTransform(params, _mapping()).transform_dict(profiles)


def _profiles(**axes: np.ndarray) -> dict:
    return {
        name: xr.DataArray(
            np.ones(len(times)), dims=["time"], coords={"time": times}, name=name
        )
        for name, times in axes.items()
    }


def test_zero_order_hold_keeps_samples_across_a_zero_crossing():
    # Tolerance used to be measured per sample, collapsing to ~1e-16 near t=0.
    step = 2.5e-4
    n = 2801
    times = (
        np.float32(-0.1) + np.arange(n).astype(np.float32) * np.float32(step)
    ).astype(np.float32)
    params = InterpolationParams(start=-0.1, step=step, method="zero")

    out = DatasetInterpolationTransform(None, _mapping()).interpolate_dimension(
        _series(times), "time", params
    )

    assert np.array_equal(out.x.values[:n], np.arange(n, dtype=float))


def test_zero_order_hold_keeps_samples_on_a_float64_axis():
    # float64 axes used to be skipped by the dtype guard.
    step = 2.5e-4
    n = 2801
    times = -0.1 + np.arange(n) * step
    params = InterpolationParams(start=-0.1, step=step, method="zero")
    assert not np.array_equal(times, np.round(times, 12))

    out = DatasetInterpolationTransform(None, _mapping()).interpolate_dimension(
        _series(times), "time", params
    )

    assert np.array_equal(out.x.values[:n], np.arange(n, dtype=float))


def test_merge_aligns_axes_that_differ_only_by_rounding():
    # Same 145 Thomson pulses, two ways of building the axis. Used to merge
    # into 283 half-NaN rows.
    period = np.float32(4.1666e-3)
    stepwise = (period * np.arange(1, 146, dtype=np.float32)).astype(np.float32)
    accumulated = np.cumsum(np.full(145, period, dtype=np.float32)).astype(np.float32)
    assert (stepwise != accumulated).sum() > 100, "fixture must actually disagree"

    merged = _merge(_profiles(t_e=stepwise, t_e_core=accumulated))

    assert merged.sizes["time"] == 145
    assert not merged.t_e.isnull().any()
    assert not merged.t_e_core.isnull().any()


def test_merge_keeps_axes_that_genuinely_differ():
    merged = _merge(_profiles(x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 20)))

    assert merged.sizes["time"] == 28  # the union, exactly as before


def test_merge_keeps_every_pulse_in_burst_mode():
    # Pulses bunch up, so the gap to the wrong neighbour shrinks.
    intra, gap = 1e-5, 1e-2
    true_times = np.array([b * gap + k * intra for b in range(18) for k in range(8)])
    stepwise = true_times.astype(np.float32)
    accumulated = np.cumsum(np.diff(true_times, prepend=0.0).astype(np.float32)).astype(
        np.float32
    )

    merged = _merge(_profiles(a=stepwise, b=accumulated))

    assert merged.sizes["time"] == np.unique(stepwise).size
    assert not merged.a.isnull().any()
    assert not merged.b.isnull().any()
