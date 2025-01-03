from abc import ABC
from typing import Union
import warnings
import scipy.signal
import numpy as np
import xarray as xr
from src.core.registry import Registry
from src.core.model import (
    DatasetInfo,
    GlobalInterpolateParams,
    InterpolationParams,
    Dimension,
    FillOptions,
)

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
        return xr.merge(datasets.values())

    def transform_dataset(self, dataset: xr.Dataset) -> xr.Dataset:
        transform_datasets = {}
        for name, channel in dataset.data_vars.items():
            dataset = self.transform_array(name, channel)
            transform_datasets[name] = dataset

        dataset = xr.merge(transform_datasets.values())
        return dataset

    def transform_array(self, signal_name: str, signal: xr.DataArray):
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )


class DatasetInterpolationTransform(BaseDatasetTransform):
    def __init__(
        self, dataset_params: DatasetInfo, global_params: GlobalInterpolateParams
    ):
        self.global_params = global_params
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

        if params.start is None and self.global_params.tmin is None:
            params.start = float(dataset.coords[dim_name].values.min())
        elif params.start is None and self.global_params.tmin is not None:
            params.start = float(self.global_params.tmin)

        if params.end is None and self.global_params.tmax is None:
            params.end = dataset.coords[dim_name].values.max()
        elif params.end is None and self.global_params.tmax is not None:
            params.end = self.global_params.tmax

        if dim_name in self.global_params.params:
            global_param_dict = self.global_params.params[dim_name].model_dump()
            for name, value in global_param_dict.items():
                if getattr(params, name) is None:
                    setattr(params, name, value)

        if params.fill == FillOptions.FFILL:
            dataset = dataset.ffill(dim=dim_name)
        elif params.fill == FillOptions.BFILL:
            dataset = dataset.bfill(dim=dim_name)

        if params.dropna:
            dataset = dataset.dropna(dim=dim_name, how="all")

        # Edge case, if the method is none, then return the data unmodified.
        if params.method == "none":
            return dataset

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            dataset = dataset.interp({dim_name: params.coords}, method=params.method)

        return dataset

    def interpolate_dimensions(
        self,
        dataset: xr.Dataset,
        dimensions: dict[str, Dimension],
        interpolate_params: dict[str, InterpolationParams] = None,
    ) -> xr.Dataset:
        for dim_name in dimensions.keys():
            if interpolate_params is not None and dim_name in interpolate_params:
                dataset = self.interpolate_dimension(
                    dataset, dim_name, interpolate_params[dim_name]
                )
            elif self.global_params is None:
                pass
            elif dim_name in self.global_params.params:
                interpolate_params = self.global_params.params[dim_name]
                dataset = self.interpolate_dimension(
                    dataset, dim_name, interpolate_params
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
        angles.attrs["units"] = "radians"
        spectrum = xr.DataArray(spectrum, coords=dict(frequency=f, time=t))

        signal = xr.Dataset(
            {f"{signal_name}_spectrogram": spectrum, f"{signal_name}_angles": angles}
        )
        return signal


transform_registry = Registry[BaseDatasetTransform]()
transform_registry.register("fftdecompose", FFTDecomposeTransform)
