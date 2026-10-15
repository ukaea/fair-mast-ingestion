"""Integration checks: level 2 output against the raw UDA signals.

These tests run the full level 2 pipeline (read, transform and write to Zarr)
for real shots. Then they read the file and compare each value with the UDA
signal that it comes from.

The tests need a connection to the UDA server, thus they are skipped on CI
(when the `CI` environment variable is set) and when UDA is not available.
To run only these tests:

    uv run pytest -m integration tests/integration
"""

import itertools
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.level2.main import process_shot
from src.level2.transforms import ZERO_HOLD_TOLERANCE

ROOT = Path(__file__).parents[2]
CONFIG_FILE = ROOT / "configs" / "level2.yml"
MAST = ROOT / "mappings" / "level2" / "mast.yml"
MASTU = ROOT / "mappings" / "level2" / "mastu.yml"


def _uda_client():
    try:
        import pyuda

        client = pyuda.Client()
        client.get("ip", 30420)
        return client
    except Exception:  # noqa: BLE001 - any failure means UDA is not available
        return None


def _skip_reason() -> str | None:
    if os.environ.get("CI"):
        return "Integration tests against UDA do not run on CI"
    if _uda_client() is None:
        return "Pyuda client unavailable"
    return None


SKIP_REASON = _skip_reason()

pytestmark = [
    pytest.mark.integration,
    pytest.mark.skipif(SKIP_REASON is not None, reason=SKIP_REASON or ""),
]


@dataclass(frozen=True)
class Case:
    mapping: Path
    shot: int
    group: str
    name: str
    source: str
    # None means: use the time dimension of the signal
    time_source: str | None = None
    # None means: use the dimension of the signal that is not time
    radius_source: str | None = None
    has_radius: bool = True
    # True when the mapping keeps the native time samples
    native_time: bool = False

    @property
    def id(self) -> str:
        return f"{self.mapping.stem}-{self.shot}-{self.group}-{self.name}"


def _thomson_mastu(shot: int, divertor: bool = True) -> list[Case]:
    groups = [("thomson_scattering", "ayc")]
    if divertor:
        groups.append(("thomson_scattering_divertor", "ayd"))
    cases = []
    for group, src in groups:
        for name in ["t_e", "n_e", "p_e"]:
            cases.append(
                Case(
                    MASTU, shot, group, name, f"{src}/{name}",
                    time_source=f"{src}/time", radius_source=f"{src}/r",
                    native_time=True,
                )
            )
    for name in ["t_e_core", "n_e_core"]:
        cases.append(
            Case(
                MASTU, shot, "thomson_scattering", name, f"ayc/{name}",
                time_source="ayc/time", has_radius=False, native_time=True,
            )
        )
    return cases


def _equilibrium_mastu(shot: int) -> list[Case]:
    sources = {"psi": "poloidalFlux", "j_phi": "jphi"}
    return [
        Case(
            MASTU, shot, "equilibrium", name, f"epm/output/profiles2D/{source}",
            time_source="epm/time", radius_source="epm/output/profiles2D/r",
        )
        for name, source in sources.items()
    ]


def _charge_exchange_mastu(shot: int) -> list[Case]:
    sources = {"t_i": "temperature", "v_i": "velocity"}
    return [
        Case(
            MASTU, shot, "charge_exchange", name, f"act/cel3/ss/pvb/c5291/{source}",
            time_source="act/cel3/ss/time", radius_source="/act/cel3/ss/major_radius",
        )
        for name, source in sources.items()
    ]


CASES = [
    *_thomson_mastu(52625),
    # The laser interval jitters in 53000. This shot has no divertor data.
    *_thomson_mastu(53000, divertor=False),
    *_equilibrium_mastu(52625),
    *_charge_exchange_mastu(52625),
    *[
        Case(MAST, 30420, "thomson_scattering", name, f"AYC_{name[0].upper()}E",
             radius_source="AYC_R")
        for name in ["t_e", "n_e", "p_e"]
    ],
    Case(MAST, 30420, "equilibrium", "psi", "EFM_PSI(R,Z)", radius_source="EFM_GRID(R)"),
    Case(MAST, 30420, "equilibrium", "j_phi", "EFM_PLASMA_CURR(R,Z)",
         radius_source="EFM_GRID(R)"),
    Case(MAST, 29880, "charge_exchange", "t_i", "ACT_SS_TEMPERATURE"),
    Case(MAST, 29880, "charge_exchange", "v_i", "ACT_SS_VELOCITY"),
]


@pytest.fixture(scope="module")
def client():
    return _uda_client()


@pytest.fixture(scope="module")
def level2_file(tmp_path_factory):
    """Write each (mapping, shot) once, with all the groups that the cases use."""
    files = {}

    def _get(mapping: Path, shot: int) -> Path:
        key = (mapping, shot)
        if key not in files:
            output = tmp_path_factory.mktemp(f"{mapping.stem}_{shot}")
            groups = sorted(
                {c.group for c in CASES if c.mapping == mapping and c.shot == shot}
            )
            process_shot(
                shot,
                mapping_file=str(mapping),
                config_file=str(CONFIG_FILE),
                output_path=str(output),
                include_datasets=groups,
                exclude_datasets=[],
                dt=0.00025,
                verbose=False,
                force_ip_check=False,
                skip_geometry=True,
            )
            files[key] = output / f"{shot}.zarr"
        return files[key]

    return _get


def _as_float(values) -> np.ndarray:
    values = np.array(values, dtype=float)
    values[~np.isfinite(values)] = np.nan
    return values


def _raw_signal(client, case: Case):
    signal = client.get(case.source, case.shot)
    data = _as_float(signal.data)

    if case.time_source is not None:
        time = np.asarray(client.get(case.time_source, case.shot).data, dtype=float)
    else:
        time = np.asarray(signal.time.data, dtype=float)
    time = time.ravel()

    radius = None
    if case.radius_source is not None:
        radius = np.asarray(client.get(case.radius_source, case.shot).data, dtype=float)
    elif case.has_radius:
        radius = next(
            np.asarray(dim.data, dtype=float)
            for dim in signal.dims
            if dim.label != signal.time.label
        )
    if radius is not None and radius.ndim > 1:
        radius = radius[0]
    return data, time, radius


def _time_index(level2_time: np.ndarray, raw_time: np.ndarray):
    """Give the raw sample that a zero order hold puts at each level 2 time."""
    step = float(np.diff(level2_time).mean())
    tolerance = step * ZERO_HOLD_TOLERANCE
    query = np.minimum(level2_time + tolerance, raw_time.max())
    index = np.searchsorted(raw_time, query, side="right") - 1
    inside = (index >= 0) & (level2_time <= raw_time.max() + tolerance)
    return index, inside


def _matches(expected: np.ndarray, actual: np.ndarray) -> bool:
    return np.allclose(expected, actual, rtol=1e-6, atol=0, equal_nan=True)


@pytest.mark.parametrize("case", CASES, ids=[c.id for c in CASES])
def test_level2_matches_uda(case: Case, client, level2_file):
    path = level2_file(case.mapping, case.shot)
    dataset = xr.open_zarr(path, group=case.group)
    assert case.name in dataset, f"{case.name} not written to {case.group}"

    raw, raw_time, radius = _raw_signal(client, case)
    order = None if radius is None else np.argsort(radius)

    # Put the level 2 data in the order (other dims, major_radius, time)
    dims = [d for d in dataset[case.name].dims if d not in ("major_radius", "time")]
    if radius is not None:
        dims.append("major_radius")
    level2 = dataset[case.name].transpose(*dims, "time")
    level2_values = _as_float(level2.values)

    if radius is not None:
        assert np.allclose(level2.major_radius.values, radius[order]), (
            "major_radius is not the native radius of the UDA signal"
        )

    if case.native_time:
        assert np.array_equal(level2.time.values, raw_time), (
            "time is not the native time of the UDA signal"
        )
    index, inside = _time_index(level2.time.values, raw_time)
    assert np.isnan(level2_values[..., ~inside]).all(), (
        "values are set at times outside the UDA signal"
    )

    # The UDA axis order is not always the same as the level 2 axis order
    for perm in itertools.permutations(range(raw.ndim)):
        expected = raw.transpose(perm)
        if expected.shape[-1] != raw_time.size:
            continue
        if order is not None:
            if expected.ndim < 2 or expected.shape[-2] != radius.size:
                continue
            expected = expected[..., order, :]
        if expected.shape[:-1] != level2_values.shape[:-1]:
            continue
        if _matches(expected[..., index[inside]], level2_values[..., inside]):
            return
    pytest.fail(f"{case.id}: level 2 values do not match the UDA signal {case.source}")
