"""Synthetic signals that carry UDA's time-base defects, for use in CI.

UDA computes time axes as ``start + n * step`` rather than storing them, so two
signals describing the same instants can be rendered by different arithmetic
routes and disagree in the last bits. These builders reproduce that mechanism
without a UDA connection, so the resampling and merge behaviour can be tested
anywhere.

Nothing here is a fixture: they are plain functions so a test can state exactly
which defect it is exercising.
"""

import numpy as np
import xarray as xr

# A reference shot, shaped like MAST 30421.
PULSE_PERIOD = 1 / 240  # Thomson YAG laser, s
N_PULSES = 145
N_RADII = 130
R_MIN, R_MAX = 0.234, 1.517


def true_times(n=N_PULSES, period=PULSE_PERIOD, start=None):
    """The instants a diagnostic actually sampled, in exact float64."""
    start = period if start is None else start
    return start + np.arange(n, dtype=np.float64) * period


def _is_uniform(times):
    if times.size < 3:
        return True
    steps = np.diff(times)
    return bool(np.allclose(steps, steps[0], rtol=1e-6, atol=0))


def axis_multiplied(times, dtype=np.float32):
    """``start + n * step`` evaluated once per sample, UDA's usual route.

    Only a uniformly sampled axis is built that way; anything else (a burst, a
    gap) is simply rendered in the narrower dtype.
    """
    times = np.asarray(times, dtype=np.float64)
    if not _is_uniform(times):
        return times.astype(dtype)
    step = np.diff(times)[0] if times.size > 1 else 0.0
    i = np.arange(times.size).astype(dtype)
    return (np.asarray(times[0], dtype) + i * np.asarray(step, dtype)).astype(dtype)


# Measured from UDA: AYC_TE and AYC_TE_CORE disagree by 61-65 float32 steps of
# the axis reach on every divergent shot sampled (3.67e-06..3.87e-06 s over
# n=122..149). The disagreement is a rendering difference and does NOT grow with
# sample count -- correlation with n is +0.21. Model it that way, or a fixture
# built for a 20 kHz shot invents an error UDA never produces.
OBSERVED_ULPS = 65


def axis_rendered(times, ulps=OBSERVED_ULPS, dtype=np.float32, seed=0):
    """The same instants rendered by UDA's other route, as measured.

    Offsets each label by up to `ulps` float32 steps of the axis reach, which is
    what the real AYC pair does, rather than compounding per sample.
    """
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=np.float64)
    step = float(np.spacing(dtype(np.abs(times).max())))
    offsets = rng.uniform(-ulps * step, ulps * step, times.size)
    return (times + offsets).astype(dtype)


def axis_accumulated(times, dtype=np.float32):
    """A pessimistic worst case: the same instants summed step by step.

    Error compounds with sample count, which real UDA does not do. Kept for
    stress tests; use `axis_rendered` when modelling the archive.
    """
    times = np.asarray(times, dtype=np.float64)
    steps = np.concatenate([[times[0]], np.diff(times)]).astype(dtype)
    return np.cumsum(steps).astype(dtype)


def axis_offset(times, offset=4.2e-12, seed=0):
    """A float64 axis sitting slightly off a decimal grid, as stored axes do."""
    rng = np.random.default_rng(seed)
    times = np.asarray(times, dtype=np.float64)
    return times + rng.uniform(-offset, offset, times.size)


def divergence(a, b):
    """Largest disagreement between two renderings of the same instants."""
    n = min(len(a), len(b))
    return float(np.abs(np.asarray(a[:n], float) - np.asarray(b[:n], float)).max())


def signal_1d(times, name="t_e_core", units="eV", values=None):
    data = np.arange(len(times), dtype=float) if values is None else values
    return xr.DataArray(
        data, dims=["time"], coords={"time": times}, name=name, attrs={"units": units}
    )


def signal_2d(times, name="t_e", units="eV", n_radii=N_RADII, values=None):
    radii = np.linspace(R_MIN, R_MAX, n_radii, dtype=np.float32)
    if values is None:
        values = np.tile(np.arange(len(times), dtype=float)[:, None], (1, n_radii))
    return xr.DataArray(
        values,
        dims=["time", "radial_index"],
        coords={"time": times, "radial_index": radii},
        name=name,
        attrs={"units": units},
    )


def radius_signal(times, n_radii=N_RADII):
    radii = np.linspace(R_MIN, R_MAX, n_radii, dtype=np.float32)
    return xr.DataArray(
        np.tile(radii, (len(times), 1)),
        dims=["time", "radial_index"],
        coords={"time": times, "radial_index": radii},
        name="AYC_R",
        attrs={"units": "m"},
    )


def thomson_loader(divergent=True, drop_pulse=None):
    """A fake ``UDALoader.load`` for the Thomson group.

    ``divergent`` renders the core and profile axes by the two different routes,
    the way UDA does. ``drop_pulse`` removes one pulse from the core signal, to
    exercise a sibling that is genuinely missing a sample.
    """
    t = true_times()
    profile_axis = axis_multiplied(t)
    core_axis = axis_accumulated(t) if divergent else profile_axis.copy()
    core_values = np.arange(N_PULSES, dtype=float)
    if drop_pulse is not None:
        core_axis = np.delete(core_axis, drop_pulse)
        core_values = np.delete(core_values, drop_pulse)

    def load(shot, name, channels=None):
        if name == "AYC_TE_CORE":
            return signal_1d(core_axis, values=core_values)
        if name == "AYC_TE":
            return signal_2d(profile_axis)
        if name == "AYC_R":
            return radius_signal(profile_axis)
        raise AssertionError(f"unexpected signal {name}")

    return load


# --- data-quality invariants -------------------------------------------------
#
# Assertions about the delivered data rather than about the code that made it.
# A pipeline's damaging failures are the ones that look plausible, so these are
# written to be reusable from both the synthetic and the live-UDA suites.


def time_axis_is_sound(times, rtol=2.0):
    """Is this a credible sampling axis? Returns a list of problems, empty if fine.

    Strictly increasing, no duplicate labels, and no step *shorter* than the
    median by more than `rtol`. A step shorter than the diagnostic's own period
    is not a measurement, it is a label that should not be there: the doubled
    Thomson axis has a median/min ratio of 279,360.

    Steps *longer* than the median are deliberately not flagged. Real MAST
    Thomson runs in burst mode on some shots -- 30310, 30318 and 30322 cluster
    pulses at 10-20 kHz and leave 33 ms between bursts, giving a max/median of
    660 with a median/min of 1.0. Those gaps are physics, not a defect.
    """
    times = np.asarray(times, dtype=np.float64)
    problems = []
    if times.size < 2:
        return problems
    if not np.all(np.diff(times) > 0):
        problems.append("time axis is not strictly increasing")
    if np.unique(times).size != times.size:
        problems.append("time axis has duplicate labels")
    steps = np.diff(times)
    median = float(np.median(steps))
    if median > 0 and steps.min() > 0 and median / steps.min() > rtol:
        problems.append(
            f"time steps as short as {steps.min():.3e} s against a median of "
            f"{median:.3e} s ({median / steps.min():.0f}x) -- spurious labels"
        )
    return problems


# Measured from UDA, not specified: see docs/time-base-methodology.md.
# ATM (ruby TS, shots <= 22831) ran at a flat 200 Hz in 21 of 21 shots sampled.
# AYC (YAG TS) runs in several modes -- 210, 240, 500, 10k and 20k Hz appear in
# 64 shots sampled from 30300-30474 -- so only a range can be asserted there.
SAMPLING_RATES_HZ = {
    "ATM": (199.9, 200.1),
    "AYC": (200.0, 25000.0),
}


def expected_rate_band(shot):
    """Which Thomson system was running, and at what rate, for this shot."""
    return SAMPLING_RATES_HZ["ATM" if shot <= 22831 else "AYC"]


def empty_time_rows(dataset):
    """Variables with time rows that are entirely NaN, and how many each has."""
    counts = {}
    for name in dataset.data_vars:
        array = dataset[name]
        dims = [d for d in array.dims if str(d).startswith("time")]
        if not dims:
            continue
        others = [d for d in array.dims if d != dims[0]]
        rows = array.isnull().all(dim=others) if others else array.isnull()
        empty = int(rows.sum())
        if empty:
            counts[name] = empty
    return counts
