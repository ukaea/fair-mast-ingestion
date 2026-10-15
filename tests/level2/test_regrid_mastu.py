"""Regridding checks for each MAST-U level 2 diagnostic group.

The UDA client is mocked, so these tests run without a UDA server. The fake
signals are built from the mapping file. Their channel counts, native sample
intervals and first sample times copy the data in UDA for MAST-U shot 52625
(soft X-rays: shot 49000, because 52625 has no soft X-ray data).

The AMC/AMB magnetics and abm bolometer timebases are float32 in UDA. When the
native samples sit on the regrid lattice, float32 rounding puts some samples
slightly after their grid point, so the fakes keep that rounding.

Thomson scattering has its own tests in test_thomson_mastu.py. The visible
cameras are not regridded (method "none"), so they are not tested here.
"""

from dataclasses import dataclass, field, replace
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pint_xarray  # noqa: F401 - registers the .pint accessor that the reader uses
import pytest

from src.core.load import UDALoader
from src.core.model import NATIVE_STEP, load_model
from src.level2.reader import DatasetReader
from src.level2.transforms import ZERO_HOLD_TOLERANCE

MAPPING_FILE = Path(__file__).parents[2] / "mappings" / "level2" / "mastu.yml"
SHOT = 52625
TMAX = 0.2  # stands in for the plasma current end time; short to keep fakes small
FAKE_START = -0.12  # fake signals start at the first native sample after this time
FAKE_END = 0.25

AMC_DT = 0.0006133612  # 1630.36 Hz


@dataclass(frozen=True)
class Timebase:
    t0: float
    dt: float
    float32: bool = False

    def times(self) -> np.ndarray:
        # Keep the native sample lattice t0 + k * dt, but only near the window
        first = max(0, int(np.ceil((FAKE_START - self.t0) / self.dt)))
        last = int(np.floor((FAKE_END - self.t0) / self.dt))
        times = self.t0 + np.arange(first, last + 1) * self.dt
        if self.float32:
            times = times.astype(np.float32).astype(np.float64)
        return times


AMC = Timebase(-2.0, AMC_DT, float32=True)
ABM = Timebase(-2.499, 1e-4, float32=True)
XDC = Timebase(-2.035, 5e-5)
EFIT = Timebase(0.02, 5e-3)


@dataclass
class Layout:
    """Native data layout in UDA for one diagnostic group."""

    time: Timebase
    # Native timebase for each source that differs from `time`
    sources: dict[str, Timebase] = field(default_factory=dict)
    # Number of channels in UDA for each source loaded channel by channel
    channels: dict[str, int] = field(default_factory=dict)
    # Native size of each non-time dimension
    sizes: dict[str, int] = field(default_factory=dict)
    # Native coordinate values, where they matter to the regrid
    coords: dict[str, np.ndarray] = field(default_factory=dict)
    # True when every source has one native rate, so the grid step must equal it
    native_rate: bool = True
    shot: int = SHOT

    def timebase(self, source: str) -> Timebase:
        return self.sources.get(source.strip("/").lower(), self.time)


EFIT_R = np.linspace(0.06, 2.0, 65)
CX_R = np.linspace(0.7609, 1.4468, 63)
# The CX rate changes between shots, so its mapping uses `step: native`
CX_ERAS = {
    "5 ms": Timebase(-0.05, 5e-3),  # up to ~48876
    "6.27 ms": Timebase(-0.0937, 0.0062667),  # from ~48877
    "10 ms": Timebase(-0.09, 0.009997),  # from ~51650
}

LAYOUTS = {
    "equilibrium": Layout(
        time=EFIT,
        sources={
            "esm/vloop/static": Timebase(0.05, 5e-3),
            "esm/vloop/dynamic": Timebase(0.05, 5e-3),
        },
        channels={
            "EPM_XPOINT_R": 2,
            "EPM_XPOINT_Z": 2,
            "EPM_STRIKE_R": 4,
            "EPM_STRIKE_Z": 4,
        },
        sizes={
            "profile_r": 65,
            "psi_norm": 65,
            "major_radius": 65,
            "z": 65,
            "n_boundary_coords": 361,
            "n_pf_current": 64,
            "n_b_field_pol_probe": 354,
            "n_flux_loop": 102,
        },
        coords={
            "major_radius": EFIT_R,
            "z": np.linspace(-2.2, 2.2, 65),
            "psi_norm": np.linspace(0.0, 1.0, 65),
        },
    ),
    "pulse_schedule": Layout(
        time=Timebase(-0.01495, 2e-4),
        sources={
            "xdc/fuelling/t/density_ref_target": Timebase(0.15, 2e-4),
            "xdc/zmanage/t/zref": Timebase(-1.0, 2e-4),
            "xdc/shape/t/rx_ref": Timebase(0.1, 2e-4),
            "xdc/shape/t/zx_ref": Timebase(0.1, 2e-4),
        },
    ),
    "summary": Layout(
        time=AMC,
        sources={
            "esm/density/nebar": Timebase(0.05, 5e-3),
            "esm/density/ng": Timebase(0.05, 5e-3),
            "esm/density/fg": Timebase(0.05, 5e-3),
            "esm/power/pohm": Timebase(0.05, 5e-3),
            "anb/sum/power": Timebase(-9.949, 5e-5),
            "anu/fc/detector1/neutrons_var": Timebase(-0.09773, 9.77e-5),
            "abm/prad": ABM,
        },
        native_rate=False,  # a summary of signals sampled from 20 kHz to 200 Hz
    ),
    "bolometer": Layout(time=ABM, sizes={"bolometer_channel": 32}),
    "interferometer": Layout(time=Timebase(-0.1, 2e-5)),
    "controllers": Layout(time=XDC),
    "gas_injection": Layout(
        time=XDC,
        sources={"xdc/gas/t": Timebase(-1.0, 2e-4)},
        channels={"XDC/GAS/F": 48, "XDC/GAS/T": 48},
        native_rate=False,  # targets at 5 kHz, measurements at 20 kHz
    ),
    "pf_active": Layout(
        time=AMC,
        sources={"xcm": Timebase(-2.5, 5e-5), "xdc/pf": XDC},
        channels={"AMC": 23, "AMC_EFCC": 4, "AMC_ELM": 12, "XCM": 13, "XDC/PF": 5},
        native_rate=False,  # AMC currents at 1.6 kHz, voltages at 20 kHz
    ),
    "pf_passive": Layout(time=AMC, channels={"AMC_CASECURR": 20}),
    "tf": Layout(time=AMC),
    "magnetics": Layout(
        time=AMC,
        sources={"xmo/omaha": Timebase(-2.5, 5e-6)},
        channels={
            "AMB": 204,
            "ASM/SAD/OUT": 19,
            "XMO/OMAHA": 19,
            "AMB_FL": 95,
            "AMB_LOOPV": 95,
            "AMC_HALO": 38,
        },
    ),
    "mse": Layout(time=EFIT, sizes={"mse_channel": 28}),
    "spectrometer_visible": Layout(time=Timebase(-0.1, 1e-5), channels={"XIM_DA": 3}),
    "charge_exchange": Layout(
        time=CX_ERAS["10 ms"],
        sizes={"major_radius": CX_R.size},
        coords={"major_radius": CX_R},
    ),
    "soft_x_rays": Layout(
        time=Timebase(-0.1, 2e-6),
        channels={"XSX/HCAM/L": 14, "XSX/HCAM/U": 14},
        shot=49000,
    ),
    "wall": Layout(time=EFIT, sizes={"limiter_geometry_channel": 98}),
}

TIME_REGRIDDED = [name for name in LAYOUTS if name not in ("mse", "wall")]
NATIVE_RATE = [name for name in TIME_REGRIDDED if LAYOUTS[name].native_rate]


def _dim(label, data, units):
    return SimpleNamespace(label=label, data=np.asarray(data), units=units)


def _signal(data, dims, units):
    data = np.asarray(data, dtype=float)
    return SimpleNamespace(
        data=data,
        errors=np.zeros_like(data),
        dims=dims,
        units=units,
        description="fake signal",
        label="fake",
    )


def _sample_ids(shape):
    """Values that change at every time index (axis 0), so each identifies its sample."""
    n_time, n_other = shape[0], int(np.prod(shape[1:], dtype=int))
    values = (np.arange(n_time)[:, None] + 1.0) + 1e-4 * np.arange(n_other)[None, :]
    return values.reshape(shape)


def _source(profile, shot):
    sources = profile.source
    if isinstance(sources, str):
        return SimpleNamespace(name=sources, channels=None)
    if len(sources) == 1:
        return sources[0]
    return next(
        s for s in sources if s.shot_range.shot_min <= shot <= s.shot_range.shot_max
    )


def _is_time(dim_name):
    return dim_name.startswith("time")


class FakeUDAClient:
    """Serves fake signals with the layout of each group in `LAYOUTS`."""

    def __init__(self, mapping, layouts=None):
        self.signals = {}
        for dataset_name, layout in (layouts or LAYOUTS).items():
            for profile in mapping.datasets[dataset_name].profiles.values():
                if profile.geometry is None:
                    self._add_profile(layout, profile)

    def _add(self, name, signal):
        self.signals[name.strip("/").lower()] = signal

    def _add_profile(self, layout, profile):
        source = _source(profile, layout.shot)
        units = profile.units or profile.target_units or ""
        time = layout.timebase(source.name).times()
        time_dim = _dim("Time", time, "s")

        if source.channels:
            for channel in source.channels:
                self._add(
                    channel.name, _signal(_sample_ids((time.size,)), [time_dim], units)
                )
            return

        dims = []
        for dim_name, dim in profile.dimensions.items():
            dim_source = dim.source if dim is not None else None
            if _is_time(dim_name):
                dims.append(time_dim)
                if dim_source:
                    self._add(dim_source, _signal(time, [time_dim], "s"))
                continue

            size = layout.sizes[dim_name]
            coord = layout.coords.get(dim_name, np.arange(1.0, size + 1))
            coord_dim = _dim(dim_name, coord, "m" if "radius" in dim_name else "")
            dims.append(coord_dim)
            if dim_source:
                self._add(dim_source, _signal(coord, [coord_dim], coord_dim.units))

        shape = [d.data.size for d in dims]
        # Put the time axis first so each sample keeps its own value
        order = sorted(
            range(len(dims)), key=lambda i: not _is_time(dims[i].label.lower())
        )
        values = np.moveaxis(
            _sample_ids([shape[i] for i in order]), range(len(order)), order
        )
        self._add(source.name, _signal(values, dims, units))

    def get(self, name, shot):
        import pyuda

        key = name.strip("/").lower()
        if key not in self.signals:
            raise pyuda.ServerException(
                "The requested dataset or attribute does not exist: check name and case"
            )
        return self.signals[key]


@pytest.fixture(scope="module")
def mapping():
    mapping = load_model(str(MAPPING_FILE))
    mapping.tmax = TMAX
    return mapping


@pytest.fixture
def fake_client(mocker, mapping):
    client = FakeUDAClient(mapping)
    mocker.patch.object(UDALoader, "_get_client", return_value=client)
    return client


@pytest.fixture
def reader(fake_client, mapping):
    return DatasetReader(mapping, UDALoader(), skip_geometry=True)


def _read(reader, dataset_name):
    dataset = reader.read_dataset(LAYOUTS[dataset_name].shot, dataset_name)
    assert len(dataset.data_vars) > 0, f"{dataset_name} read no profiles"
    return dataset


def _profiles(mapping, dataset_name):
    for name, profile in mapping.datasets[dataset_name].profiles.items():
        if profile.geometry is None:
            yield name, profile


def _time_dims(profile):
    return [name for name in profile.dimensions if _is_time(name)]


@pytest.mark.parametrize("dataset_name", LAYOUTS)
def test_channel_counts_match_uda(reader, mapping, dataset_name):
    layout = LAYOUTS[dataset_name]
    dataset = _read(reader, dataset_name)
    interpolate = mapping.datasets[dataset_name].interpolate or {}

    for name, profile in _profiles(mapping, dataset_name):
        source = _source(profile, layout.shot)
        for dim_name in profile.dimensions:
            regridded = (
                dim_name in interpolate and interpolate[dim_name].method != "none"
            )
            if _is_time(dim_name) or regridded:
                continue
            if source.channels:
                expected = layout.channels[source.name]
                assert len(source.channels) == expected, (
                    f"{name}: mapping lists {len(source.channels)} channels for "
                    f"{source.name}, UDA has {expected}"
                )
            else:
                expected = layout.sizes[dim_name]
            assert dataset[name].sizes[dim_name] == expected, f"{name}.{dim_name}"


@pytest.mark.parametrize("dataset_name", NATIVE_RATE)
def test_time_step_matches_native_rate(reader, mapping, dataset_name):
    layout = LAYOUTS[dataset_name]
    dataset = _read(reader, dataset_name)

    for name, profile in _profiles(mapping, dataset_name):
        source = _source(profile, layout.shot)
        native_dt = layout.timebase(source.name).dt
        for dim_name in _time_dims(profile):
            step = np.diff(dataset[dim_name].values)
            assert np.allclose(step, native_dt, rtol=1e-6), (
                f"{name}.{dim_name}: grid step {step[0]:.6g} s, native step {native_dt:.6g} s"
            )


@pytest.mark.parametrize("dataset_name", TIME_REGRIDDED)
def test_regrid_keeps_every_native_sample(reader, mapping, dataset_name):
    layout = LAYOUTS[dataset_name]
    dataset = _read(reader, dataset_name)
    interpolate = mapping.datasets[dataset_name].interpolate

    for name, profile in _profiles(mapping, dataset_name):
        source = _source(profile, layout.shot)
        native = layout.timebase(source.name)
        for dim_name in _time_dims(profile):
            params = interpolate[dim_name]
            if params.method == "none":
                continue
            step = native.dt if params.step == NATIVE_STEP else params.step
            if step > native.dt * (1 + 1e-6):
                continue  # a coarser grid must drop samples; summary does this on purpose

            out = (
                dataset[name]
                .transpose(dim_name, ...)
                .values.reshape(dataset.sizes[dim_name], -1)
            )
            column = out[:, np.isfinite(out).sum(axis=0).argmax()]
            held = np.unique(column[np.isfinite(column)])

            grid = dataset[dim_name].values
            raw = native.times()
            expected = ((raw >= grid[0]) & (raw <= grid[-1])).sum()
            assert held.size >= expected, (
                f"{name}.{dim_name}: {expected} native samples in the window, "
                f"only {held.size} kept"
            )


def test_equilibrium_radial_grid_is_native_efit_grid(reader):
    dataset = _read(reader, "equilibrium")

    assert np.allclose(dataset.major_radius.values, EFIT_R)


def test_charge_exchange_radial_grid_covers_chords(reader):
    dataset = _read(reader, "charge_exchange")
    radius = dataset.major_radius.values

    assert radius.min() <= CX_R.min()
    assert radius.max() >= CX_R.max()
    assert np.diff(radius).max() <= np.diff(CX_R).min()
    # Most of the grid has data, so it is not padded far past the chords
    has_data = dataset.t_i.notnull().any("time").values
    assert has_data.mean() > 0.8


@pytest.mark.parametrize("era", CX_ERAS)
def test_charge_exchange_grid_follows_shot_rate(mocker, mapping, era):
    layout = replace(LAYOUTS["charge_exchange"], time=CX_ERAS[era])
    client = FakeUDAClient(mapping, {"charge_exchange": layout})
    mocker.patch.object(UDALoader, "_get_client", return_value=client)
    reader = DatasetReader(mapping, UDALoader(), skip_geometry=True)

    dataset = reader.read_dataset(SHOT, "charge_exchange")
    grid = dataset.time.values
    raw = layout.time.times()

    assert np.allclose(np.diff(grid), layout.time.dt, rtol=1e-6)
    held = np.unique(dataset.t_i_core.values[np.isfinite(dataset.t_i_core.values)])
    # A sample on the last grid point counts, within the hold tolerance
    end = grid[-1] + layout.time.dt * ZERO_HOLD_TOLERANCE
    assert held.size == ((raw >= grid[0]) & (raw <= end)).sum()
