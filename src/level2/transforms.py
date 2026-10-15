import math
import warnings
from abc import ABC
from typing import Optional, Union

import numpy as np
import scipy.signal
import xarray as xr

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

# Fraction of a grid step within which a sample counts as on the grid point
ZERO_HOLD_TOLERANCE = 1e-3


class BaseDatasetTransform(ABC):
    def transform(self, data: Union[ProfileDict, xr.Dataset]) -> xr.Dataset:
        if isinstance(data, dict):
            return self.transform_dict(data)
        elif isinstance(data, xr.Dataset):
            return self.transform_dataset(data)

    def transform_dict(self, profiles: ProfileDict) -> xr.Dataset:
        datasets = {}
        for profile_name, profile in profiles.items():
            profile = self.transform_array(profile_name, profile)
            datasets[profile_name] = profile
        ds = xr.merge(datasets.values())
        ds.attrs = dict() #clear unwanted attributes at dataset level
        return ds

    def transform_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        transform_datasets = {}
        for name, channel in dataset.data_vars.items():
            dataset = self.transform_array(name, channel)
            transform_datasets[name] = dataset
        ds = xr.merge(transform_datasets.values())
        ds.attrs = dict() #clear unwanted attributes at dataset level
        return ds

    def transform_array(self, signal_name: str, signal: xr.DataArray):
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )


class DatasetInterpolationTransform(BaseDatasetTransform):
    def __init__(
        self, dataset_params: DatasetInfo, mapping: Mapping
    ):
        self.mapping = mapping
        self.dataset_params = dataset_params
        self._interpolate = None if dataset_params is None else dataset_params.interpolate

    def transform(self, data: Union[ProfileDict, xr.Dataset]) -> xr.Dataset:
        if isinstance(data, dict):
            profiles = data
        else:
            profiles = {str(name): array for name, array in data.data_vars.items()}
        self._interpolate = self._resolve_native_steps(profiles)
        if isinstance(data, dict):
            data = self._align_native_coords(data)
        return super().transform(data)

    def _align_native_coords(self, profiles: ProfileDict) -> ProfileDict:
        """Put the profiles on one shared axis for each dimension not interpolated.

        Profiles can read the same samples with small floating point differences.
        The merge then keeps the near-equal values as different points, and each
        profile has NaN at the points of the other profiles.
        """
        if self._interpolate is None:
            return profiles
        for dim_name, params in self._interpolate.items():
            if params.method == "none":
                profiles = self._snap_to_shared_axis(profiles, dim_name)
        return profiles

    @staticmethod
    def _snap_to_shared_axis(profiles: ProfileDict, dim_name: str) -> ProfileDict:
        coords = {
            name: profile[dim_name]
            for name, profile in profiles.items()
            if dim_name in profile.coords
            and profile[dim_name].ndim == 1
            and profile.sizes[dim_name] > 0
            and np.issubdtype(profile[dim_name].dtype, np.number)
        }
        if not coords:
            return profiles

        # The profile with the most points sets the shared axis
        longest = max(coords.values(), key=lambda c: c.size)
        shared = np.sort(longest.values.astype(float))
        steps = np.diff(shared)
        steps = steps[steps > 0]
        if steps.size == 0:
            return profiles
        tolerance = float(np.median(steps)) * ZERO_HOLD_TOLERANCE

        aligned = dict(profiles)
        for name, coord in coords.items():
            values = coord.values.astype(float)
            index = np.clip(np.searchsorted(shared, values), 1, shared.size - 1)
            left, right = shared[index - 1], shared[index]
            nearest = np.where(values - left <= right - values, left, right)
            snapped = np.where(np.abs(nearest - values) <= tolerance, nearest, values)
            if np.unique(snapped).size != snapped.size:
                logger.warning(
                    f"Cannot align '{dim_name}' of profile '{name}': two points "
                    f"are within {tolerance} of the same shared point."
                )
                continue
            if not np.array_equal(snapped, values):
                logger.debug(
                    f"Aligned '{dim_name}' of profile '{name}' by up to "
                    f"{np.abs(snapped - values).max()}"
                )
                aligned[name] = profiles[name].assign_coords(
                    {dim_name: coord.copy(data=snapped)}
                )
            # Points that match no shared point join the shared axis
            shared = np.union1d(shared, snapped)
        return aligned

    def transform_array(self, signal_name: str, signal: xr.DataArray):
        dimensions = self.dataset_params.profiles[signal_name].dimensions
        profile = self.interpolate_dimensions(signal, dimensions, self._interpolate)
        return profile

    def _resolve_native_steps(
        self, profiles: ProfileDict
    ) -> Optional[dict[str, InterpolationParams]]:
        """Replace each `step: native` with the native step of the data.

        All profiles in a dataset share one grid, so the step is the smallest
        native step of the profiles that have the dimension.
        """
        interpolate = self.dataset_params.interpolate
        if interpolate is None:
            return None

        resolved = {}
        for dim_name, params in interpolate.items():
            if params.step == NATIVE_STEP:
                step = self._native_step(profiles, dim_name)
                logger.debug(f"Native step for dimension '{dim_name}' is {step}")
                params = params.model_copy(update={"step": step})
            resolved[dim_name] = params
        return resolved

    @staticmethod
    def _native_step(profiles: ProfileDict, dim_name: str) -> float:
        steps = []
        for profile in profiles.values():
            if dim_name not in profile.dims or profile.sizes[dim_name] < 2:
                continue
            coord = profile[dim_name].values
            # Mean step, so a grid with this step does not drift from the samples
            steps.append(float(coord[-1] - coord[0]) / (coord.size - 1))
        if not steps:
            raise ValueError(
                f"Cannot find the native step for dimension '{dim_name}': "
                f"no profile has two or more points on it."
            )
        return min(steps)

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
            raise ValueError(
                f"Dimension '{dim_name}' has `step: native`, which only "
                f"`transform` can resolve from the data."
            )

        coords = self._build_aligned_grid(
            params.start, params.end, params.step, end_was_explicit
        )
        query = coords
        if params.method == "zero":
            # UDA timebases are often float32, so a sample on a grid point can
            # read as slightly after it. Without a tolerance the hold then takes
            # the previous sample, which duplicates one sample and drops the next.
            tolerance = params.step * ZERO_HOLD_TOLERANCE
            query = coords + tolerance
            # The same applies to the last sample, but interp gives NaN past the
            # end of the data, so clip grid points on the last sample onto it.
            last = dataset.coords[dim_name].values.max()
            query = np.where(coords <= last + tolerance, np.minimum(query, last), query)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = dataset.interp({dim_name: query}, method=params.method)
        dataset = dataset.assign_coords({dim_name: coords})
        return dataset

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
        return start + np.arange(n + 1) * step

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
        time_dim = next((dim for dim in data.dims if dim == "time" or dim.startswith("time_")), None)
        time_dim_str = str(time_dim)
        if not time_dim:
            logger.warning(f"Skipping background subtraction: No time dimension found in dataset {data.name}.")
            return data
        isel_kwargs = {time_dim: slice(self.start, self.end)}
        background = data.isel(**isel_kwargs).mean(dim=time_dim_str)
        return data - background

transform_registry = Registry[BaseDatasetTransform]()
transform_registry.register("fftdecompose", FFTDecomposeTransform)