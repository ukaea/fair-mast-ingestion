"""Does level 2 resampling preserve the samples the source actually took?

These are our own correctness checks. They do not need UDA: the time-base
defects UDA produces are reproduced synthetically in `fake_signals`, so CI can
run every one of them. The defect itself is UDA's problem and is reported
separately; what is tested here is that our resampling and merge survive it.
"""

import numpy as np
import pint  # noqa: F401  registers the .pint accessor the reader uses
import pint_xarray  # noqa: F401
import pytest
import xarray as xr

from src.core.load import UDALoader
from src.core.model import (
    DatasetInfo,
    Dimension,
    InterpolationParams,
    Mapping,
    ProfileInfo,
)
from src.level2.reader import DatasetReader
from src.level2.transforms import DatasetInterpolationTransform

from fake_signals import (
    N_PULSES,
    axis_rendered,
    PULSE_PERIOD,
    empty_time_rows,
    expected_rate_band,
    time_axis_is_sound,
    axis_accumulated,
    axis_multiplied,
    axis_offset,
    divergence,
    thomson_loader,
    true_times,
)


def _mapping() -> Mapping:
    return Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="summary/ip",
        datasets={},
    )


def _transform(params=None) -> DatasetInterpolationTransform:
    return DatasetInterpolationTransform(params, _mapping())


def _series(times, values=None) -> xr.Dataset:
    """Values equal their own index, so a stale hold is obvious."""
    if values is None:
        values = np.arange(len(times), dtype=float)
    return xr.DataArray(
        values, dims=["time"], coords={"time": times}, name="x"
    ).to_dataset()


def _merge(**axes) -> xr.Dataset:
    profiles = {
        name: xr.DataArray(
            np.ones(len(times)), dims=["time"], coords={"time": times}, name=name
        )
        for name, times in axes.items()
    }
    params = DatasetInfo(
        profiles={
            name: ProfileInfo(source=name, dimensions={"time": None})
            for name in profiles
        }
    )
    return _transform(params).transform_dict(profiles)


def _thomson_mapping() -> Mapping:
    radius = Dimension(source="AYC_R")
    return Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="summary/ip",
        datasets={
            "thomson_scattering": DatasetInfo(
                # no `time` block: Thomson is deliberately not time-interpolated
                interpolate={
                    "major_radius": InterpolationParams(
                        start=0.3, end=1.49, step=0.01, method="linear", fill="none"
                    )
                },
                profiles={
                    "t_e": ProfileInfo(
                        source="AYC_TE",
                        dimensions={"time": None, "major_radius": radius},
                        dimension_order=["major_radius", "time"],
                    ),
                    "t_e_core": ProfileInfo(
                        source="AYC_TE_CORE", dimensions={"time": None}
                    ),
                },
            )
        },
    )


# --- the fake data must actually carry the defect ------------------------------


def test_fixture_axes_diverge():
    t = true_times()
    a, b = axis_multiplied(t), axis_accumulated(t)

    assert (a != b).sum() > 100, "the two routes must disagree"
    # same order as the real AYC pair: microseconds, on a 4.2 ms period
    assert 1e-7 < divergence(a, b) < 1e-5
    assert divergence(a, b) < PULSE_PERIOD / 100


def test_fixture_float64_off_grid():
    t = true_times(n=500)
    off = axis_offset(t)

    assert not np.array_equal(t, off)
    assert 1e-13 < divergence(t, off) < 1e-10


# --- do we keep every sample the source took? ---------------------------------


@pytest.mark.parametrize(
    "dtype,build",
    [
        (np.float32, axis_multiplied),
        (np.float32, axis_accumulated),
        (np.float64, lambda t: axis_offset(t)),
        (np.float64, lambda t: np.asarray(t, dtype=np.float64)),
    ],
    ids=["f32-multiplied", "f32-accumulated", "f64-offset", "f64-exact"],
)
def test_sample_survives_resampling(dtype, build):
    """The core claim: resampling must not lose or misplace a real sample."""
    step = 2.5e-4
    n = 2001
    t = -0.1 + np.arange(n) * step
    axis = build(t) if dtype is np.float64 else build(t, dtype)
    params = InterpolationParams(start=-0.1, step=step, method="zero")

    out = _transform().interpolate_dimension(_series(axis), "time", params)

    assert np.array_equal(out.x.values[:n], np.arange(n, dtype=float))


def test_zero_crossing():
    """Tolerance used to be measured per sample, collapsing to ~1e-16 near t=0."""
    step = 2.5e-4
    n = 2801
    axis = axis_multiplied(-0.1 + np.arange(n) * step)
    params = InterpolationParams(start=-0.1, step=step, method="zero")

    out = _transform().interpolate_dimension(_series(axis), "time", params)

    assert np.array_equal(out.x.values[:n], np.arange(n, dtype=float))


def test_finest_grid():
    """magnetics/time_omaha is 5e-7 s, the tightest step in the mapping."""
    step = 5e-7
    n = 20001
    axis = axis_offset(-0.1 + np.arange(n) * step)
    params = InterpolationParams(start=-0.1, step=step, method="zero")

    out = _transform().interpolate_dimension(_series(axis), "time", params)

    assert np.array_equal(out.x.values[:n], np.arange(n, dtype=float))


def test_source_finer_than_grid():
    source = np.round(-0.1 + np.arange(2001) * 5e-5, 12)
    coarse = np.round(-0.1 + np.arange(101) * 1e-3, 12)

    out = DatasetInterpolationTransform._snap_coord_to_grid(
        _series(source), "time", coarse
    )

    assert np.unique(out.time.values).size == source.size


# --- sampling rate: are we matching the right samples together? ---------------


def test_rate_match_same_instants():
    t = true_times()
    assert DatasetInterpolationTransform._same_sampling(
        axis_accumulated(t), axis_multiplied(t).astype(np.float64)
    )


def test_rate_match_missing_pulse():
    t = true_times()
    short = np.delete(axis_accumulated(t), 70)
    assert DatasetInterpolationTransform._same_sampling(
        short, axis_multiplied(t).astype(np.float64)
    )


@pytest.mark.parametrize("factor", [2.0, 0.5, 1.5])
def test_rate_match_rejects_other_rate(factor):
    t = true_times()
    other = true_times(n=int(N_PULSES / factor), period=PULSE_PERIOD * factor)
    assert not DatasetInterpolationTransform._same_sampling(other, t)


def test_merge_keeps_rates_apart():
    merged = _merge(x=np.linspace(0, 1, 10), y=np.linspace(0, 1, 20))

    assert merged.sizes["time"] == 28  # the union, exactly as before


# --- merging siblings ---------------------------------------------------------


def test_merge_aligns_siblings():
    t = true_times()

    merged = _merge(t_e=axis_multiplied(t), t_e_core=axis_accumulated(t))

    assert merged.sizes["time"] == N_PULSES
    assert not merged.t_e.isnull().any()
    assert not merged.t_e_core.isnull().any()


def test_merge_pads_missing_pulse():
    t = true_times()
    short = np.delete(axis_accumulated(t), 70)

    merged = _merge(t_e=axis_multiplied(t), t_e_core=short)

    assert merged.sizes["time"] == N_PULSES
    assert int(merged.t_e_core.isnull().sum()) == 1
    assert not merged.t_e.isnull().any()


@pytest.mark.parametrize("intra", [1e-3, 1e-4, 1e-5, 1e-6])
def test_merge_burst_rates(intra):
    """Burst mode: pulses bunch up, so the gap to the wrong neighbour shrinks."""
    t = np.array([b * 1e-2 + k * intra for b in range(18) for k in range(8)])
    a, b = axis_multiplied(t), axis_accumulated(t)

    merged = _merge(a=a, b=b)

    assert merged.sizes["time"] == np.unique(a).size
    assert not merged.a.isnull().any()
    assert not merged.b.isnull().any()


# --- the whole reader path, on a fake reference shot ---------------------------


def test_thomson_merges(mocker, reference_shot):
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())

    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)
    ds = reader.read_dataset(reference_shot, "thomson_scattering")

    assert ds.sizes["time"] == N_PULSES  # was 283, the union of both axes
    assert ds.sizes["major_radius"] == 120
    assert not ds.t_e.isnull().all(dim="major_radius").any()
    assert not ds.t_e_core.isnull().any()


def test_thomson_values_stay_put(mocker, reference_shot):
    """Alignment must not shift a profile against the core trace."""
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())

    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)
    ds = reader.read_dataset(reference_shot, "thomson_scattering")

    core = ds.t_e_core.values
    assert np.array_equal(core, np.arange(N_PULSES, dtype=float))


def test_thomson_dropped_pulse(mocker, reference_shot):
    mocker.patch(
        "src.core.load.UDALoader.load", side_effect=thomson_loader(drop_pulse=70)
    )

    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)
    ds = reader.read_dataset(reference_shot, "thomson_scattering")

    assert ds.sizes["time"] == N_PULSES
    assert int(ds.t_e_core.isnull().sum()) == 1


def test_thomson_noop_when_aligned(mocker, reference_shot):
    """The alignment must be a no-op where there is nothing to align."""
    mocker.patch(
        "src.core.load.UDALoader.load", side_effect=thomson_loader(divergent=False)
    )

    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)
    ds = reader.read_dataset(reference_shot, "thomson_scattering")

    assert ds.sizes["time"] == N_PULSES
    assert not ds.t_e_core.isnull().any()


# --- data quality: assertions on the delivered data ---------------------------
#
# The checks above prove the fix behaves. These assert properties the published
# data must hold whatever the code does, because a pipeline's damaging failures
# are the plausible-looking ones.


def test_axis_sound(mocker, reference_shot):
    """Strictly increasing, no duplicates, one population of time step."""
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())
    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)

    ds = reader.read_dataset(reference_shot, "thomson_scattering")

    assert time_axis_is_sound(ds.time.values) == []


def test_axis_check_catches_doubling():
    """Guard the guard: the pre-fix axis must fail the soundness check."""
    t = true_times()
    doubled = np.unique(
        np.concatenate(
            [
                axis_multiplied(t).astype(np.float64),
                axis_accumulated(t).astype(np.float64),
            ]
        )
    )

    problems = time_axis_is_sound(doubled)

    assert problems, "a doubled axis must be reported as unsound"
    assert "spurious labels" in problems[0]


def test_no_empty_rows(mocker, reference_shot):
    """Completeness: a doubled axis shows up as whole rows of NaN."""
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())
    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)

    ds = reader.read_dataset(reference_shot, "thomson_scattering")

    assert empty_time_rows(ds) == {}


def test_row_count(mocker, reference_shot):
    """Not merely non-zero: the axis must hold the pulses the laser fired."""
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())
    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)

    ds = reader.read_dataset(reference_shot, "thomson_scattering")
    t = ds.time.values.astype(np.float64)
    expected = round((t.max() - t.min()) / PULSE_PERIOD) + 1

    assert ds.sizes["time"] == expected == N_PULSES


def test_no_value_shift():
    """The stale-hold defect: 18 real EFIT slices carried their predecessor's values."""
    step = 2.5e-4
    n = 2001
    grid = DatasetInterpolationTransform._build_aligned_grid(
        -0.1, -0.1 + (n - 1) * step, step, True
    )
    axis = axis_multiplied(-0.1 + np.arange(n) * step)
    values = np.arange(n, dtype=float)  # value == index, so a shift is obvious
    params = InterpolationParams(start=-0.1, step=step, method="zero")

    out = _transform().interpolate_dimension(_series(axis, values), "time", params)
    held = out.x.values[: len(grid)]

    assert not np.array_equal(
        held[1:], values[:-1]
    ), "every value shifted by one sample"
    assert np.array_equal(held, values)


def test_idempotent(mocker, reference_shot):
    """Idempotence: a re-run after a fix must not change what is published."""
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())
    mapping = _thomson_mapping()

    first = DatasetReader(mapping, UDALoader(), skip_geometry=True).read_dataset(
        reference_shot, "thomson_scattering"
    )
    second = DatasetReader(mapping, UDALoader(), skip_geometry=True).read_dataset(
        reference_shot, "thomson_scattering"
    )

    assert np.array_equal(first.time.values, second.time.values)
    for name in first.data_vars:
        assert np.array_equal(
            first[name].values, second[name].values, equal_nan=True
        ), name


# --- sampling rates, historical and across modes ------------------------------


@pytest.mark.parametrize(
    "rate_hz",
    [200.0, 210.0, 240.0, 500.0, 10_000.0, 20_000.0],
    ids=["atm-200", "ayc-210", "ayc-240", "ayc-500", "ayc-10k", "ayc-20k"],
)
def test_merge_at_every_rate(rate_hz):
    """Every rate MAST Thomson has actually run at, over a realistic shot.

    Axes span 0.6 s, so the divergence is the one a full shot produces. The
    tolerance is a quarter of the sampling period, so it shrinks as the rate
    climbs -- at 20 kHz it is 1.25e-05 s against a 3.9e-06 s divergence, a
    margin of 3.2x. That holds because UDA's divergence is a fixed ~65 float32
    steps of the axis reach and does not grow with sample count.
    """
    period = 1 / rate_hz
    n = int(0.6 / period) + 1
    t = true_times(n=n, period=period, start=period)

    merged = _merge(t_e=axis_multiplied(t), t_e_core=axis_rendered(t))

    assert merged.sizes["time"] == n
    assert time_axis_is_sound(merged.time.values) == []
    assert not merged.t_e.isnull().any()
    assert not merged.t_e_core.isnull().any()


def test_merge_degrades_safely():
    """A stress case UDA does not produce: error compounding past half a step.

    When two renderings disagree by more than half the sampling interval, no
    tolerance can pair them -- a sample of one is nearer a *different* sample of
    the other. The merge must then fall back to the union it produced before,
    never fusing two pulses together.
    """
    period = 5e-5  # 20 kHz
    n = int(0.6 / period) + 1
    t = true_times(n=n, period=period, start=period)
    pessimistic = axis_accumulated(t)
    assert divergence(axis_multiplied(t), pessimistic) > period / 2

    merged = _merge(t_e=axis_multiplied(t), t_e_core=pessimistic)

    assert merged.sizes["time"] >= n, "a pulse was fused into its neighbour"


def test_burst_gaps_allowed():
    """Shots 30310/30318/30322 cluster at 20 kHz with 33 ms between bursts."""
    burst = np.array(
        [b * 33e-3 + k * 5e-5 for b in range(12) for k in range(10)], dtype=np.float64
    )
    steps = np.diff(burst)

    assert steps.max() / np.median(steps) > 300, "fixture must look like a burst"
    assert time_axis_is_sound(burst) == []


def test_short_steps_flagged():
    """The one direction that is never physics: a label inside the dead time."""
    t = true_times(n=50)
    doubled = np.unique(
        np.concatenate(
            [
                axis_multiplied(t).astype(np.float64),
                axis_accumulated(t).astype(np.float64),
            ]
        )
    )

    assert time_axis_is_sound(doubled) != []


@pytest.mark.parametrize(
    "shot,system", [(12698, "ATM"), (22831, "ATM"), (22832, "AYC"), (30421, "AYC")]
)
def test_rate_band_by_era(shot, system):
    """The band must switch where mast.yml switches source, at 22832."""
    low, high = expected_rate_band(shot)

    assert (low, high) == ((199.9, 200.1) if system == "ATM" else (200.0, 25000.0))


def test_reference_shot_rate(mocker, reference_shot):
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())
    reader = DatasetReader(_thomson_mapping(), UDALoader(), skip_geometry=True)

    ds = reader.read_dataset(reference_shot, "thomson_scattering")
    rate = 1 / float(np.median(np.diff(np.sort(ds.time.values.astype(np.float64)))))
    low, high = expected_rate_band(reference_shot)

    assert low <= rate <= high
