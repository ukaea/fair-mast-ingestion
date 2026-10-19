import math
import warnings
from abc import ABC
from typing import Union

import numpy as np
import scipy.signal
import xarray as xr

from src.core.log import logger
from src.core.model import (
    DatasetInfo,
    Dimension,
    FillOptions,
    InterpolationParams,
    Mapping,
)
from src.core.registry import Registry

ProfileDict = dict[str, xr.DataArray]


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
            ref = np.sort(np.asarray(ref, dtype=np.float64))
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

        times = coord.values.astype(np.float64)
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


class DatasetInterpolationTransform(BaseDatasetTransform):
    def __init__(self, dataset_params: DatasetInfo, mapping: Mapping):
        self.mapping = mapping
        self.dataset_params = dataset_params

    def transform_array(self, signal_name: str, signal: xr.DataArray):
        dimensions = self.dataset_params.profiles[signal_name].dimensions
        profile = self.interpolate_dimensions(
            signal, dimensions, self.dataset_params.interpolate
        )
        return profile

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

        coords = self._build_aligned_grid(
            params.start, params.end, params.step, end_was_explicit
        )
        dataset = self._snap_coord_to_grid(dataset, dim_name, coords)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = dataset.interp({dim_name: coords}, method=params.method)
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
