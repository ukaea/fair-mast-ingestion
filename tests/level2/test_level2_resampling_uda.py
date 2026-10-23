"""Level 2 resampling compared with the native UDA samples, across MAST eras.

Each test reads a group from UDA the way the reader does, resamples it the way
the reader does, and compares the result with the UDA samples it came from.
Skipped in CI and off the VPN.

Two claims are checked on every time axis:

- **values**: each published row is the UDA sample in force at that grid time
  (a zero-order hold). Nothing is invented, interpolated, or moved by more
  than one grid step. This must hold for every group and every shot.
- **loss**: every UDA sample inside the grid window appears in at least one
  row. This fails where the grid step is coarser than the native rate.
  `KNOWN_LOSS` lists those cases, measured, with the native rate. A listed case
  that no longer loses samples fails too, so the list cannot go stale.

Radial axes are checked separately: a published radius should be a radius
the diagnostic measured.
"""

import functools

import numpy as np
import pint  # noqa: F401  registers the .pint accessor the reader uses
import pint_xarray  # noqa: F401
import pytest
import xarray as xr

from src.core.load import loader_registry
from src.core.model import load_model
from src.level2.reader import DatasetReader

pytestmark = pytest.mark.uda

MAPPING = "mappings/level2/mast.yml"

ERAS = {"M5": 12698, "M6": 16146, "M7": 24600, "M8": 27880, "M9": 30421}
# Each of these takes minutes per shot (MHz rates, many channels)
HEAVY = {"magnetics", "soft_x_rays", "spectrometer_visible"}
HEAVY_ERAS = ("M5", "M7", "M9")
NOT_RESAMPLED = {"camera_visible", "wall", "thomson_scattering"}

# (group, dim, shot): why samples are lost. Measured against UDA.
KNOWN_LOSS = {
    **{
        ("summary", "time", shot): "1 kHz summary grid; ip, n_e, neutrons, NBI and "
        "P_rad are sampled at 2.5-100 kHz"
        for shot in ERAS.values()
    },
}


def _cases():
    groups = load_model(MAPPING).datasets
    for group in groups:
        if group in NOT_RESAMPLED:
            continue
        eras = HEAVY_ERAS if group in HEAVY else ERAS
        for era in eras:
            yield pytest.param(group, ERAS[era], id=f"{group}-{era}")


@pytest.fixture(scope="module")
def mapping():
    return load_model(MAPPING)


@pytest.fixture(scope="module")
def reader(mapping):
    return DatasetReader(mapping, loader_registry.create("uda"), skip_geometry=True)


@functools.lru_cache(maxsize=1)
def _read(shot, group):
    """The native profiles and the published group, from one UDA read."""
    reader = DatasetReader(
        load_model(MAPPING), loader_registry.create("uda"), skip_geometry=True
    )
    profiles = reader.read_profiles(shot, group)
    if not profiles:
        return {}, xr.Dataset()
    published = reader.apply_interpolation(dict(profiles), group)
    return profiles, published


def _plain(array):
    if array.pint.units is not None:
        return array.pint.dequantify()
    return array


def _rows(array, dim):
    """The values as a (time, everything else) float64 matrix."""
    array = array.transpose(dim, ...)
    return array.values.astype(np.float64).reshape(array.sizes[dim], -1)


def _true_times(values):
    """The sample times, without the float32 rounding of an even axis.

    Snapping raw float32 times loses samples where the native rate is near the
    grid step (AMM at 4999.2 Hz on 5 kHz), so the oracle would be wrong. Where
    the axis is within 4 float32 steps of an even axis, the even axis is used.
    """
    times = np.asarray(values, dtype=np.float64)
    if np.asarray(values).dtype != np.float32 or times.size < 3:
        return times
    even = np.linspace(times[0], times[-1], times.size)
    tol = 4 * float(np.spacing(np.float32(np.abs(times).max())))
    return even if np.abs(times - even).max() <= tol else times


def _held(native, dim, grid):
    """Which native sample a zero-order hold puts at each grid point.

    An independent oracle for the resampler: the last sample at or before each
    grid point, where "at" allows the same quarter-step snap the code uses.
    -1 marks a grid point with no sample: before the first, or after the
    first grid point at or after the last (later rows come from siblings).
    """
    times = _true_times(native[dim].values)
    # The grid step, not the mapping's: `step: native` is resolved per shot
    step = float(np.median(np.diff(grid)))
    steps = np.diff(times)
    steps = steps[steps > 0]
    tol = 0.25 * min(step, steps.min() if steps.size else step)
    index = np.searchsorted(times, grid + tol, side="right") - 1
    end = np.searchsorted(grid, times[-1] - tol)
    index[end + 1 :] = -1
    return index


def _check_time(profiles, published, mapping, group):
    """Per (profile, dim): mismatched rows, and the native samples lost."""
    info = mapping.datasets[group]
    mismatched, lost = {}, {}
    for name, native in profiles.items():
        if name not in published:
            continue  # no samples in the grid window; see the test at the end
        native = _plain(native)
        # e.g. EFIT pads the LCFS to 2600 points; the reader drops the padding
        for d, p in (info.interpolate or {}).items():
            if d in native.dims and p.dropna and p.method == "none":
                native = native.dropna(d, how="all")
        dims = [
            d
            for d, p in (info.interpolate or {}).items()
            if d in native.dims and p.method != "none"
        ]
        # Radially regridded profiles cannot be compared sample for sample
        if any(info.interpolate[d].method != "zero" for d in dims):
            continue
        for dim in dims:
            params = info.interpolate[dim]
            if params.dropna:
                native = native.dropna(dim, how="all")
            out = _plain(published[name])
            grid = out[dim].values.astype(np.float64)
            index = _held(native, dim, grid)

            want, got = _rows(native, dim), _rows(out, dim)
            expect = np.where(index[:, None] >= 0, want[np.maximum(index, 0)], np.nan)
            same = (got == expect) | (np.isnan(got) & np.isnan(expect))
            if not same.all():
                mismatched[(name, dim)] = int((~same.all(axis=1)).sum())

            times = _true_times(native[dim].values)
            inside = np.flatnonzero((times >= grid[0]) & (times <= grid[-1]))
            missing = np.setdiff1d(inside, index)
            if missing.size:
                lost[(name, dim)] = (missing.size, inside.size)
    return mismatched, lost


@pytest.mark.parametrize("group,shot", list(_cases()))
def test_time_matches_uda(mapping, group, shot):
    profiles, published = _read(shot, group)
    if not profiles:
        pytest.skip(f"{group} has no data for shot {shot}")

    mismatched, lost = _check_time(profiles, published, mapping, group)

    assert mismatched == {}, f"rows that are not the UDA sample in force: {mismatched}"

    known = {k for k in KNOWN_LOSS if k[0] == group and k[2] == shot}
    lost_dims = {(group, dim, shot) for (_, dim) in lost}
    unexpected = {k: v for k, v in lost.items() if (group, k[1], shot) not in known}
    assert unexpected == {}, f"UDA samples lost (lost, of): {unexpected}"

    checked = {dim for (_, dim) in _checked_dims(profiles, mapping, group)}
    stale = {k for k in known if k[1] in checked and k not in lost_dims}
    assert not stale, f"no longer loses samples, remove from KNOWN_LOSS: {stale}"


def _checked_dims(profiles, mapping, group):
    info = mapping.datasets[group]
    for name, native in profiles.items():
        if any(
            p.method not in ("zero", "none")
            for d, p in (info.interpolate or {}).items()
            if d in native.dims
        ):
            continue
        for dim, p in (info.interpolate or {}).items():
            if dim in native.dims and p.method == "zero":
                yield name, dim


@pytest.mark.parametrize("era", list(ERAS))
def test_bolometer_on_time(mapping, era):
    """The grid is the native ABM rate, so each sample has its own row.

    Every UDA sample must appear once, at its own time. This is the strict
    form of the loss check: no sample dropped, repeated, or delayed.
    """
    shot = ERAS[era]
    profiles, published = _read(shot, "bolometer")
    if not profiles:
        pytest.skip(f"no bolometer data for shot {shot}")

    for name in ("power", "power_radiated_total"):
        native = _plain(profiles[name]).dropna("time", how="all")
        out = _plain(published[name])
        times = native.time.values.astype(np.float64)
        grid = out.time.values.astype(np.float64)

        assert grid.size == times.size, f"{name}: one row per UDA sample"
        assert np.abs(grid - times).max() < 1e-6, f"{name}: rows at UDA times"
        assert np.array_equal(_rows(out, "time"), _rows(native, "time"), equal_nan=True)


# --- radial axes --------------------------------------------------------------


RADIAL = (
    [
        pytest.param("thomson_scattering", "t_e", shot, id=f"thomson-{era}")
        for era, shot in ERAS.items()
    ]
    + [
        pytest.param("charge_exchange", "t_i", shot, id=f"cxrs-{era}")
        for era, shot in ERAS.items()
        if era in ("M7", "M8")
    ]
    + [
        pytest.param("equilibrium", "psi", shot, id=f"efit-{era}")
        for era, shot in ERAS.items()
    ]
)


@pytest.mark.parametrize("group,profile,shot", RADIAL)
def test_radii_are_measured(mapping, group, profile, shot):
    """Every channel the diagnostic measured is published at its own radius."""
    profiles, published = _read(shot, group)
    if profile not in profiles:
        pytest.skip(f"no {group}/{profile} for shot {shot}")

    native = profiles[profile].major_radius.values.astype(np.float64)
    native = native[np.isfinite(native)]
    axis = published[profile].major_radius.values.astype(np.float64)

    distance = np.abs(native[:, None] - axis[None, :]).min(axis=1)
    assert distance.max() < 1e-6, (
        f"{(distance >= 1e-6).sum()} of {native.size} measured radii are not "
        f"published; furthest {distance.max():.4f} m"
    )


# --- one bad profile must not take the group with it -------------------------


@pytest.mark.parametrize(
    "group,shot,profile",
    [
        ("equilibrium", 28500, "vloop_static"),
        ("summary", 28500, "line_average_n_e"),
        ("gas_injection", 26399, "valve_target_voltage"),
    ],
)
def test_group_survives_empty_profile(reader, group, shot, profile):
    """These profiles have no samples after the grid start in UDA.

    EFIT writes [-1000, -999] for what it did not compute; XDC/GAS/T on 26399
    ends at -0.1015 s. Each used to raise, and the whole group was lost.
    """
    ds = reader.read_dataset(shot, group)

    assert profile not in ds.data_vars
    assert len(ds.data_vars) > 0
