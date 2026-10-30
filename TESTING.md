# Testing

## Running the tests

```bash
pytest tests/                 # everything available here
pytest tests/level2           # just level 2
CI=1 pytest tests/            # exactly what GitHub runs
pytest tests/level2 -m uda    # only the tests that need a live UDA server
pytest -rs tests/             # ...and show why anything was skipped
```

Nothing needs a UDA connection by default. Tests that do are marked `@pytest.mark.uda`
and skip themselves automatically when `CI` is set or no server answers, so a laptop
off the VPN runs the whole suite green.

```
CI=1 pytest tests/level2    52 passed, 19 skipped
pytest tests/level2         69 passed,  2 skipped    (on the VPN, ~72s)
```

`tests/conftest.py` reads the UDA variables out of `[tool.pytest_env]` in
`pyproject.toml` itself, so the live tests work whether or not the `pytest-env`
plugin happens to be installed.

## How the test suite is arranged

| path | what it is |
|---|---|
| `tests/conftest.py` | shared fixtures, the `uda` marker, the UDA availability probe |
| `tests/core/` | loaders, writers, upload |
| `tests/level1/` | level 1 loader, writer, transforms |
| `tests/level2/fake_signals.py` | **synthetic signals carrying UDA's real defects** |
| `tests/level2/test_level2_time_base.py` | time-axis correctness, runs in CI |
| `tests/level2/test_level2_time_base_uda.py` | the same claims against live UDA |
| `tests/level2/test_level2_resampling_uda.py` | every group compared with its UDA samples, M5 to M9 |
| `tests/level2/test_reader.py` | reader behaviour |
| `tests/level2/test_model.py` | mapping files parse |

## Writing a test that needs diagnostic data

Don't reach for UDA. `tests/level2/fake_signals.py` builds signals that carry the
defects real UDA data has, so the test runs in CI:

```python
from fake_signals import true_times, axis_multiplied, axis_rendered

t = true_times()                 # the instants the laser actually fired
profile = axis_multiplied(t)     # start + n*step, UDA's usual route
core = axis_rendered(t)          # the same instants, UDA's other rendering
```

`profile` and `core` describe the *same* pulses but disagree in the last bits,
exactly as `AYC_TE` and `AYC_TE_CORE` do. That disagreement is the defect most of
the level 2 time-base tests exist to handle.

Use `axis_rendered`, not `axis_accumulated`. The measured divergence is a fixed
61–65 float32 steps of the axis reach and does **not** grow with sample count;
`axis_accumulated` compounds per sample, so at 20 kHz it invents an error ten
times larger than UDA produces. It is kept only as a deliberate stress case.

For a whole diagnostic group, `thomson_loader()` is a drop-in for `UDALoader.load`:

```python
def test_something(mocker, reference_shot):
    mocker.patch("src.core.load.UDALoader.load", side_effect=thomson_loader())
    ds = DatasetReader(mapping, UDALoader(), skip_geometry=True).read_dataset(
        reference_shot, "thomson_scattering"
    )
```

Pass `drop_pulse=70` for a sibling genuinely missing a sample, or `divergent=False`
for axes that already agree.

### Assert on the data, not only on the code

A pipeline's damaging failures are the plausible-looking ones — data that is valid
but wrong. Two helpers exist for asserting on what was produced:

```python
time_axis_is_sound(ds.time.values)   # [] if the axis is credible, else reasons
empty_time_rows(ds)                  # {variable: count} of all-NaN time rows
```

`time_axis_is_sound` flags steps *shorter* than the diagnostic's own period —
a label inside the dead time is not a measurement. It deliberately does **not**
flag long gaps, because MAST Thomson runs in burst mode on some shots: 30310,
30318 and 30322 cluster pulses at 10–20 kHz and leave 33 ms between bursts.

## The UDA-marked tests

These exist to check the synthetic fixtures still describe the archive, not to
re-test the logic. The canary is explicit:

```python
assert differing > 100, "if this drops to 0, UDA has changed its axis build"
```

**If a UDA test fails while the CI suite passes, the fixtures have drifted from
reality.** Re-measure before changing any code.

To run them you need the VPN and `pytest-env` is optional:

```bash
pytest tests/level2 -m uda -rs
```

Two currently skip with an explicit reason — `magnetics` and `pf_active` hit a
separate units-mapping defect that is nothing to do with time bases.

## What the level 2 time-base tests cover

Grouped as they appear in `test_level2_time_base.py`:

- **Fixture self-checks.** The synthetic axes must still disagree the way UDA's do,
  otherwise every merge test below passes for nothing.
- **Sample survival.** Every source sample comes back at its own grid point, across
  float32 and float64, both build routes, a zero crossing, and the finest grid step
  in the mapping (5e-7 s).
- **Sampling rate.** Axes sampled at the same rate are aligned; axes at different
  rates are left alone. Covers the rates MAST has actually used — 200 Hz on ATM,
  and 210 / 240 / 500 Hz / 10 kHz / 20 kHz on AYC.
- **Merging.** Siblings rendered differently collapse onto one axis; a sibling
  missing a pulse gets one NaN row rather than doubling the axis; no rate ever
  fuses two pulses.
- **Data quality.** Axis soundness, completeness, row count against the sampling
  rate, no value attributed to the wrong sample, and idempotence.
- **The reader path.** The whole of `read_dataset` driven over mocked UDA.
- **Grid ends.** The last sample is kept when it is not on the grid. A float32
  last time does not add an empty row. A profile with no samples after the grid
  start is skipped, and the other profiles in the group are kept.
- **Float32 even axes.** An even float32 axis is rebuilt in float64 before the
  snap. Without this, AMM at 4999.2 Hz on the 5 kHz grid loses samples where the
  phase drift crosses the snap tolerance. An axis with a gap is not rebuilt.
- **Native step.** `step: native` gives one row for each sample of the fastest
  profile, at its sample time. A slower profile is held, not lost.

## Comparing every group with UDA

`test_level2_resampling_uda.py` reads each resampled group for one shot in each
of the campaigns M5 to M9 (12698, 16146, 24600, 27880, 30421). The groups
`magnetics`, `soft_x_rays` and `spectrometer_visible` are read for M5, M7 and M9
only, because each takes minutes per shot. Then the test compares the output
with the UDA samples:

- **Values.** Each published row must be the UDA sample in force at that grid
  time. The oracle is a plain zero-order hold that the test calculates itself.
  There are no exceptions.
- **Loss.** Each UDA sample in the grid window must appear in a row. The cases
  where the grid is coarser than the native rate are in `KNOWN_LOSS`, with the
  measured rate. Only `summary` is in it: its 1 kHz grid is intentional. If a
  listed case does not lose samples, the test fails, so you must remove the
  entry. Diagnostics whose rate changes between campaigns use `step: native`,
  so the oracle gets the step from the published grid.
- **Bolometer.** One row for each ABM sample, at the ABM time, with the same value.
- **Radial axes.** A published radius must be a measured radius. Thomson
  scattering, CXRS and EFIT keep their native radii (`method: none`). Linear
  interpolation onto a fixed grid flattened single-channel peaks, and cut the
  Thomson channels outside 0.3 m to 1.49 m.
- **Empty profiles.** Shots where UDA gives a profile with no samples after the
  grid start (EFIT `-999` placeholders, `XDC/GAS/T` on 26399) must still give
  the rest of the group.

A full run takes about 30 minutes on the VPN.

## Checking a test actually tests something

A test that passes for the wrong reason is worse than no test. Two habits catch it:

**Break the code and watch.** Revert the behaviour under test and confirm the
expected tests fail:

```bash
git stash push src/level2/transforms.py
CI=1 pytest tests/level2 -q        # these should now fail
git stash pop
```

**Check the fixture is still hard.** Several tests assert their own fixture carries
the defect — `test_fixture_axes_diverge`, `test_axis_check_catches_doubling`. Keep
that habit: a fixture that quietly becomes trivial takes its test with it.

**Check the fixture against the archive.** Two real mistakes from this suite, both
caught only by measuring UDA rather than trusting the model:

- the sampling-rate test built 120-sample axes, where the divergence is 2000×
  smaller than over a real 0.6 s shot, so it passed at every rate without
  exercising anything;
- the axis builder compounded error per sample, inventing a divergence at 20 kHz
  that UDA never produces, which made the merge look broken at high rates when it
  was not.

## Naming

Short names, the way the rest of the repo does it — `test_thomson_merges`, not
`test_reference_shot_thomson_merges_onto_one_axis`. The name says which behaviour;
the docstring says why it matters and what it would catch.
