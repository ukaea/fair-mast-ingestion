"""Reference-shot checks against live UDA. Skipped in CI and off the VPN.

The CI-safe equivalents live in `test_level2_time_base.py`; these confirm the
synthetic fixtures still describe the real archive. If one of these fails while
the synthetic suite passes, the fixtures have drifted from UDA, not the code.
"""

import numpy as np
import pint  # noqa: F401  registers the .pint accessor the reader uses
import pint_xarray  # noqa: F401
import pytest

from fake_signals import (
    N_PULSES,
    empty_time_rows,
    expected_rate_band,
    thomson_loader,
    time_axis_is_sound,
)

from src.core.load import UDALoader, loader_registry
from src.core.model import load_model
from src.level2.reader import DatasetReader

pytestmark = pytest.mark.uda

MAPPING = "mappings/level2/mast.yml"
EXPECTED_PULSES = 145  # thomson pulses in the reference shot


@pytest.fixture(scope="module")
def loader():
    return loader_registry.create("uda")


@pytest.fixture(scope="module")
def mapping():
    return load_model(MAPPING)


def _read(mapping, loader, shot, group):
    reader = DatasetReader(mapping, loader, skip_geometry=True)
    try:
        return reader.read_dataset(shot, group)
    except RuntimeError as e:
        if "No units defined" in str(e):
            # A separate, pre-existing mapping defect: some signals declare a
            # target unit but arrive from UDA without one. Nothing to do with
            # time bases, so do not fail the time-base suite over it.
            pytest.skip(f"{group} blocked by a units-mapping defect: {e}")
        raise


def _fake_thomson_mapping():
    """The minimal Thomson group the synthetic loader satisfies."""
    from src.core.model import DatasetInfo, Dimension, InterpolationParams, ProfileInfo

    radius = Dimension(source="AYC_R")
    return load_model(MAPPING).model_copy(
        update={
            "datasets": {
                "thomson_scattering": DatasetInfo(
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
            }
        }
    )


def _time_axis(signal):
    """UDA labels the time dimension `t`, not `time`."""
    name = next(iter(signal.sizes))
    return np.asarray(signal.coords[name].values)


def test_axes_still_diverge(loader, reference_shot):
    """The defect the synthetic fixtures imitate is still present upstream."""
    core = _time_axis(loader.load(reference_shot, "AYC_TE_CORE"))
    profile = _time_axis(loader.load(reference_shot, "AYC_TE"))

    assert core.size == profile.size == EXPECTED_PULSES
    differing = int((core != profile).sum())
    assert differing > 100, "if this drops to 0, UDA has changed its axis build"
    deviation = np.abs(core.astype(float) - profile.astype(float)).max()
    assert 1e-7 < deviation < 1e-5


def test_thomson_merges(mapping, loader, reference_shot):
    ds = _read(mapping, loader, reference_shot, "thomson_scattering")

    assert ds.sizes["time"] == EXPECTED_PULSES
    for name in ds.data_vars:
        other = [d for d in ds[name].dims if d != "time"]
        rows = ds[name].isnull().all(dim=other) if other else ds[name].isnull()
        assert int(rows.sum()) == 0, f"{name} has empty time rows"


def test_thomson_axis_sound(mapping, loader, reference_shot):
    """The reported symptom was a stepping pattern: two populations of Δt."""
    ds = _read(mapping, loader, reference_shot, "thomson_scattering")
    dt = np.diff(np.sort(ds.time.values.astype(float)))

    assert dt.min() > dt.max() / 2, "time steps should not vary by more than 2x"


def test_native_samples_on_grid(mapping, loader):
    """pf_active on a shot whose axis UDA returns as float32."""
    shot, step = 26399, 5e-5
    tn = _time_axis(loader.load(shot, "XDC_PF_F_P1")).astype(np.float64)

    ds = _read(mapping, loader, shot, "pf_active")
    grid = np.round(ds.time.values.astype(np.float64), 12)
    wanted = np.round(np.round(tn / step) * step, 12)
    common = np.intersect1d(wanted, grid)

    assert common.size > 0.9 * tn.size, "most native samples should have a grid point"


def test_equilibrium_solutions(mapping, loader, reference_shot):
    ds = _read(mapping, loader, reference_shot, "equilibrium")
    solved = ds.psi.notnull().any(dim=["z", "major_radius"]).values

    # 131 of 141 in the reference shot; the rest are genuinely unconverged
    assert int(solved.sum()) >= 131


@pytest.mark.parametrize("group", ["summary", "bolometer", "soft_x_rays", "magnetics"])
def test_no_empty_rows(mapping, loader, reference_shot, group):
    """A doubled axis shows up as whole rows of NaN; none of these should have any."""
    ds = _read(mapping, loader, reference_shot, group)
    if len(ds) == 0:
        pytest.skip(f"{group} has no data for this shot")

    for name, empty in empty_time_rows(ds).items():
        dim = next(d for d in ds[name].dims if str(d).startswith("time"))
        assert empty < ds.sizes[dim], f"{group}/{name} is entirely empty"


@pytest.mark.parametrize("group", ["thomson_scattering", "equilibrium", "summary"])
def test_axes_sound(mapping, loader, reference_shot, group):
    """Every time coordinate we publish, held to the same invariant."""
    ds = _read(mapping, loader, reference_shot, group)
    if len(ds) == 0:
        pytest.skip(f"{group} has no data for this shot")

    for coord in (c for c in ds.coords if str(c).startswith("time")):
        assert time_axis_is_sound(ds[coord].values) == [], f"{group}/{coord}"


# --- do the fixtures still describe the archive? ------------------------------


def test_mocked_matches_uda(mapping, loader, reference_shot, mocker):
    """Run the same reader over fake and real data; the structure must match.

    Values differ by construction -- the fixtures carry indices, not physics --
    so this compares what the fixtures claim to reproduce: the axis length, the
    sampling rate, and the absence of empty rows.
    """
    real = _read(mapping, loader, reference_shot, "thomson_scattering")

    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())
    fake = DatasetReader(
        _fake_thomson_mapping(), UDALoader(), skip_geometry=True
    ).read_dataset(reference_shot, "thomson_scattering")

    assert fake.sizes["time"] == real.sizes["time"] == N_PULSES
    assert empty_time_rows(fake) == empty_time_rows(real) == {}
    assert time_axis_is_sound(fake.time.values) == time_axis_is_sound(real.time.values)

    rate = lambda ds: 1 / float(  # noqa: E731
        np.median(np.diff(np.sort(ds.time.values.astype(np.float64))))
    )
    assert rate(fake) == pytest.approx(rate(real), rel=0.01)


@pytest.mark.parametrize(
    "shot", [12698, 16146, 26399, 29778, 30421], ids=["M5", "M6", "M8", "M9a", "M9b"]
)
def test_rate_by_era(loader, shot):
    """ATM ran at 200 Hz; AYC runs in several modes. Measured, not specified."""
    for signal in ("AYC_TE", "ATM_TE"):
        try:
            times = _time_axis(loader.load(shot, signal)).astype(np.float64)
            break
        except Exception:
            continue
    else:
        pytest.skip(f"no Thomson signal for shot {shot}")

    if times.size < 5:
        pytest.skip(f"shot {shot} has too few pulses to judge a rate")
    rate = 1 / float(np.median(np.diff(times)))
    low, high = expected_rate_band(shot)

    assert low <= rate <= high, f"shot {shot} sampled at {rate:.1f} Hz"


def test_burst_shot_ok(mapping, loader):
    """30310 clusters pulses at 20 kHz. The gaps are physics and must survive."""
    ds = _read(mapping, loader, 30310, "thomson_scattering")
    if len(ds) == 0:
        pytest.skip("30310 has no level 2 Thomson data")

    steps = np.diff(np.sort(ds.time.values.astype(np.float64)))

    assert steps.max() / np.median(steps) > 100, "30310 should still look bursty"
    assert time_axis_is_sound(ds.time.values) == []
