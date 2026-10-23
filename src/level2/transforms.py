import math
import warnings
from abc import ABC
from typing import Optional, TypeVar, Union

import numpy as np
import scipy.signal
import xarray as xr

from src.core.load import MissingProfileError
from src.core.log import logger
from src.core.model import (
    NATIVE_STEP,
    DatasetInfo,
    Dimension,
    FillOptions,
    InterpolationParams,
    Mapping,
)
from src.core.registry import Registry

ProfileDict = dict[str, xr.DataArray]
Xarray = TypeVar("Xarray", xr.Dataset, xr.DataArray)


class BaseDatasetTransform(ABC):
    def transform(self, data: Union[ProfileDict, xr.Dataset]) -> xr.Dataset:
        if isinstance(data, dict):
            return self.transform_dict(data)
        elif isinstance(data, xr.Dataset):
            return self.transform_dataset(data)

    def transform_dict(self, profiles: ProfileDict) -> xr.Dataset:
        datasets = {}
        for profile_name, profile in profiles.items():
            try:
                profile = self.transform_array(profile_name, profile)
            except MissingProfileError as e:
                # One empty profile must not take the rest of the group with it.
                logger.warning(f"Skipping profile {profile_name}: {e}")
                continue
            datasets[profile_name] = profile
        datasets = self._unify_shared_axes(datasets)
        ds = xr.merge(datasets.values())
        ds.attrs = dict()  # clear unwanted attributes at dataset level
        return ds

    def _unify_shared_axes(self, profiles: ProfileDict) -> ProfileDict:
        """Snap sibling profiles' axes onto the longest one before merging.

        UDA builds each signal's axis separately, so two signals covering the
        same samples disagree in the last bits and `xr.merge` unions them into a
        half-NaN axis. Samples the reference does not have are left alone and
        still outer-join, so a sibling genuinely missing a pulse merges with a
        NaN on that row rather than doubling the axis.
        """
        profiles = dict(profiles)
        for dim in {dim for array in profiles.values() for dim in array.dims}:
            members = [
                name
                for name, array in profiles.items()
                if dim in array.coords and array.coords[dim].dtype.kind == "f"
            ]
            if len(members) < 2:
                continue
            ref = max((profiles[n].coords[dim].values for n in members), key=len)
            ref = np.sort(self._float64_axis(ref))
            for name in members:
                if self._same_sampling(profiles[name].coords[dim].values, ref):
                    profiles[name] = self._snap_coord_to_grid(profiles[name], dim, ref)
        return profiles

    @staticmethod
    def _same_sampling(times: np.ndarray, ref: np.ndarray, rtol: float = 0.01) -> bool:
        """Were these two axes sampled at the same rate?

        The precondition for "these are the same instants, rounded differently".
        Without it, two axes genuinely sampled at different rates have labels
        that fall close enough to snap together, which would claim two separate
        measurements happened at one instant.
        """
        a, b = np.diff(np.sort(np.asarray(times, dtype=np.float64))), np.diff(ref)
        a, b = a[a > 0], b[b > 0]
        if not a.size or not b.size:
            return False
        return abs(np.median(a) - np.median(b)) <= rtol * np.median(b)

    def transform_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        transform_datasets = {}
        for name, channel in dataset.data_vars.items():
            dataset = self.transform_array(name, channel)
            transform_datasets[name] = dataset
        ds = xr.merge(transform_datasets.values())
        ds.attrs = dict()  # clear unwanted attributes at dataset level
        return ds

    def transform_array(self, signal_name: str, signal: xr.DataArray):
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )

    @staticmethod
    def _snap_coord_to_grid(
        dataset: xr.Dataset, dim_name: str, grid: np.ndarray, frac: float = 0.25
    ) -> xr.Dataset:
        """Convert a float coordinate to float64 and snap it to the grid.

        A plain cast keeps the rounding error (0.07 becomes 0.0700000003), and
        zero-order-hold interpolation then drops the sample as later than its
        own grid point. A time within `frac` of a step of its nearest grid point
        is moved onto it. `grid` is an interpolation grid, or a sibling axis.

        The tolerance is a fraction of a step, not of float precision. UDA
        computes axes rather than storing them, so two renderings of the same
        instants sit whole arithmetic routes apart.
        """
        coord = dataset.coords[dim_name]
        if coord.dtype.kind != "f":
            return dataset
        grid = np.asarray(grid, dtype=np.float64)
        if grid.size < 2:
            return dataset

        times = BaseDatasetTransform._float64_axis(coord.values)
        spacings = [float(np.diff(grid).min())]
        own = np.diff(np.sort(times))
        own = own[own > 0]
        if own.size:
            spacings.append(float(own.min()))
        tol = frac * min(spacings)

        idx = np.clip(np.searchsorted(grid, times), 1, grid.size - 1)
        left, right = grid[idx - 1], grid[idx]
        nearest = np.where(np.abs(times - left) <= np.abs(right - times), left, right)
        times = np.where(np.abs(times - nearest) <= tol, nearest, times)
        return dataset.assign_coords({dim_name: (coord.dims, times, coord.attrs)})

    @staticmethod
    def _float64_axis(values: np.ndarray, ulps: float = 4) -> np.ndarray:
        """Convert an axis to float64, and rebuild a float32 even axis exactly.

        UDA gives many axes as float32 renderings of `t0 + k * dt`. Each label
        is then up to one float32 step from its true time. At 1 MHz and above
        that is a large fraction of a sample, and where the native rate is near
        the grid step, the snap tolerance cannot tell the jitter from the true
        phase, so samples are lost. If the float32 axis is within `ulps` float32
        steps of an even axis between its end points, the even axis is used.
        """
        times = np.asarray(values, dtype=np.float64)
        if np.asarray(values).dtype != np.float32 or times.size < 3:
            return times
        even = np.linspace(times[0], times[-1], times.size)
        tol = ulps * float(np.spacing(np.float32(np.abs(times).max())))
        if np.abs(times - even).max() <= tol:
            return even
        return times


class DatasetInterpolationTransform(BaseDatasetTransform):
    def __init__(self, dataset_params: DatasetInfo, mapping: Mapping):
        self.mapping = mapping
        self.dataset_params = dataset_params
        self._interpolate = (
            None if dataset_params is None else dataset_params.interpolate
        )

    def transform(self, data: Union[ProfileDict, xr.Dataset]) -> xr.Dataset:
        if isinstance(data, dict):
            profiles = data
        else:
            profiles = {str(name): array for name, array in data.data_vars.items()}
        self._interpolate = self._resolve_native_steps(profiles)
        return super().transform(data)

    def transform_array(self, signal_name: str, signal: xr.DataArray):
        dimensions = self.dataset_params.profiles[signal_name].dimensions
        profile = self.interpolate_dimensions(signal, dimensions, self._interpolate)
        return profile

    def _resolve_native_steps(
        self, profiles: ProfileDict
    ) -> Optional[dict[str, InterpolationParams]]:
        """Replace each `step: native` with the native step of the data.

        All profiles in a dataset share one grid, so the profile with the
        smallest native step sets it. The grid start moves forward onto a
        sample time of that profile, so each of its samples has its own row.
        """
        interpolate = self.dataset_params.interpolate
        if interpolate is None:
            return None

        resolved = {}
        for dim_name, params in interpolate.items():
            if params.step == NATIVE_STEP:
                start = (
                    params.start
                    if params.start is not None
                    else self.mapping.default_start
                )
                grid = self._native_grid(profiles, dim_name, start)
                if grid is not None:
                    step, start = grid
                    logger.debug(f"Native grid for '{dim_name}': {start} + k * {step}")
                    params = params.model_copy(update={"step": step, "start": start})
            resolved[dim_name] = params
        return resolved

    @classmethod
    def _native_grid(
        cls, profiles: ProfileDict, dim_name: str, start: Optional[float]
    ) -> Optional[tuple[float, Optional[float]]]:
        """The smallest native step on a dimension, and the grid start for it.

        None if no profile has two or more increasing points on the dimension,
        e.g. a shot before the diagnostic was installed.
        """
        best = None
        for profile in profiles.values():
            if dim_name not in profile.dims or profile.sizes[dim_name] < 2:
                continue
            times = cls._float64_axis(profile[dim_name].values)
            if not np.all(np.diff(times) > 0):
                continue
            # The mean step of an even axis. The median of a float32 axis is
            # quantised (10 kHz reads as 9998 Hz), so the grid would drift.
            step = (times[-1] - times[0]) / (times.size - 1)
            even = np.abs(times - np.linspace(times[0], times[-1], times.size))
            if even.max() > 0.25 * step:
                # Gaps or bursts: the mean is not a step that occurs
                step = float(np.median(np.diff(times)))
            if best is None or step < best[0]:
                best = (float(step), float(times[0]))
        if best is None:
            return None
        step, first = best
        if start is None:
            return step, None
        # The first sample time at or after `start`, extended back if the
        # data starts later than the grid. A sample within the snap tolerance
        # before `start` counts as on it (float32 -0.1 is -0.1000000015).
        k = math.ceil((start - first) / step - 0.25)
        return step, first + k * step

    def interpolate_dimension(
        self, dataset: xr.Dataset, dim_name: str, params: InterpolationParams
    ) -> xr.Dataset:
        params = params.model_copy()
        end_was_explicit = params.end is not None

        if not end_was_explicit:
            if self.mapping.tmax is not None:
                params.end = float(self.mapping.tmax)
            else:
                params.end = float(dataset.coords[dim_name].values.max())
            start = (
                params.start if params.start is not None else self.mapping.default_start
            )
            if start is not None and params.end < start:
                # e.g. EFIT writes [-1000, -999] for a quantity it did not compute
                raise MissingProfileError(
                    f"No samples on '{dim_name}' after the grid start {start} "
                    f"(last sample at {params.end})"
                )

        if params.fill == FillOptions.FFILL:
            dataset = dataset.ffill(dim=dim_name)
        elif params.fill == FillOptions.BFILL:
            dataset = dataset.bfill(dim=dim_name)

        if params.dropna:
            dataset = dataset.dropna(dim=dim_name, how="all")

        # Edge case, if the method is none, then return the data unmodified.
        if params.method == "none":
            return dataset

        if params.start is None:
            if self.mapping.default_start is None:
                raise ValueError(
                    f"Dimension '{dim_name}' has no `start` set, and "
                    f"`default_start` is not configured in the mapping. "
                    f"Set one in the dataset's `interpolate` block or "
                    f"add `default_start` to the top of the mapping YAML."
                )
            params.start = self.mapping.default_start

        if params.step is None:
            raise ValueError(
                f"Dimension '{dim_name}' has no `step` in the dataset's "
                f"`interpolate` block — cannot infer a default."
            )
        if params.step == NATIVE_STEP:
            # `transform` resolves it, unless no profile has two samples on it
            raise MissingProfileError(
                f"No native step for '{dim_name}': no profile in the dataset "
                f"has two or more increasing samples on it"
            )

        coords = self._build_aligned_grid(
            params.start, params.end, params.step, end_was_explicit
        )
        source = self._snap_coord_to_grid(dataset, dim_name, coords)
        if not end_was_explicit and self.mapping.tmax is None:
            # The grid was sized from the unsnapped last time. A float32 last
            # time a little after its grid point adds a point that has no data.
            last = source[dim_name].values.max()
            coords = coords[: np.searchsorted(coords, last) + 1]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = source.interp({dim_name: coords}, method=params.method)
        if params.method == "zero":
            dataset = self._hold_last_sample(source, dataset, dim_name, params.step)
        return dataset

    @staticmethod
    def _hold_last_sample(
        source: Xarray, resampled: Xarray, dim_name: str, step: float
    ) -> Xarray:
        """Hold the last sample at the grid point that follows it.

        The grid ends up to one step after the last sample, but the zero-order
        hold gives NaN after the last sample. Without this, the last sample of
        every signal that is not on the grid is lost.
        """
        times = source[dim_name].values
        if times.size == 0:
            return resampled
        last = int(np.argmax(times))
        grid = resampled[dim_name].values
        tail = np.flatnonzero((grid > times[last]) & (grid < times[last] + step))
        if tail.size == 0:
            return resampled
        held = source.isel({dim_name: last}, drop=True)
        if isinstance(resampled, xr.DataArray):
            resampled.loc[{dim_name: grid[tail]}] = held
            return resampled
        for name, array in resampled.data_vars.items():
            if dim_name in array.dims:
                array.loc[{dim_name: grid[tail]}] = held[name]
        return resampled

    @staticmethod
    def _build_aligned_grid(
        start: float, end: float, step: float, end_was_explicit: bool
    ) -> np.ndarray:
        if step <= 0:
            raise ValueError(f"Interpolation step must be positive, got {step}")
        if end < start:
            raise ValueError(f"end ({end}) must be >= start ({start})")

        bin_float_tol = 1e-9
        extent_in_bins = (end - start) / step
        if end_was_explicit:
            n = math.floor(extent_in_bins + bin_float_tol)
        else:
            n = math.ceil(extent_in_bins - bin_float_tol)
        n = max(n, 0)
        # Remove accumulated float error (e.g. -0.1 + 31 * 0.005 is not exactly
        # 0.055), so grid points match decimal source times exactly.
        return np.round(start + np.arange(n + 1) * step, 12)

    def interpolate_dimensions(
        self,
        dataset: xr.Dataset,
        dimensions: dict[str, Dimension],
        interpolate_params: dict[str, InterpolationParams] = None,
    ) -> xr.Dataset:
        if interpolate_params is None:
            return dataset
        for dim_name in dimensions.keys():
            if dim_name in interpolate_params:
                dataset = self.interpolate_dimension(
                    dataset, dim_name, interpolate_params[dim_name]
                )
        return dataset


class FFTDecomposeTransform(BaseDatasetTransform):
    def __init__(self, nperseg: int = 256):
        self.nperseg = nperseg

    def transform_array(self, signal_name: str, signal: xr.DataArray) -> xr.Dataset:
        data = signal.values
        times = signal.time.values

        freq = 1 / (times[1] - times[0])
        f, t, X = scipy.signal.stft(data, fs=freq, nperseg=self.nperseg, noverlap=0)

        f = f[: self.nperseg // 2]
        f /= 1000

        f = xr.DataArray(f, name="frequency", dims=["frequency"])
        f.attrs["units"] = "Hz"

        t = xr.DataArray(t, name="time", dims=["time"])
        t.attrs["untis"] = "s"

        spectrum = X[: self.nperseg // 2]

        angles = np.angle(spectrum)
        spectrum = np.abs(spectrum)

        angles = xr.DataArray(angles, coords=dict(frequency=f, time=t))
        angle_name = f"{signal_name}_angles"
        angles.attrs["name"] = angle_name
        angles.attrs["units"] = "radians"

        spectrum = xr.DataArray(spectrum, coords=dict(frequency=f, time=t))
        spec_name = f"{signal_name}_spectrogram"
        spectrum.attrs["name"] = spec_name

        signal = xr.Dataset({spec_name: spectrum, angle_name: angles})
        return signal


class BackgroundSubtractionTransform(BaseDatasetTransform):
    def __init__(self, start: int, end: int):
        self.start = start
        self.end = end

    def transform_array(self, data: xr.DataArray) -> xr.DataArray:
        # subtracts background calculated from mean of data points between given start and end
        time_dim = next(
            (dim for dim in data.dims if dim == "time" or dim.startswith("time_")), None
        )
        time_dim_str = str(time_dim)
        if not time_dim:
            logger.warning(
                f"Skipping background subtraction: No time dimension found in dataset {data.name}."
            )
            return data
        isel_kwargs = {time_dim: slice(self.start, self.end)}
        background = data.isel(**isel_kwargs).mean(dim=time_dim_str)
        return data - background


transform_registry = Registry[BaseDatasetTransform]()
transform_registry.register("fftdecompose", FFTDecomposeTransform)
