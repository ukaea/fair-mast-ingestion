import re
from abc import ABC
from enum import Enum
from typing import Optional

import fsspec
import numpy as np
import xarray as xr
import zarr
import zarr.storage

from src.registry import Registry


class MissingProfileError(Exception):
    pass


class MissingSourceError(Exception):
    pass


class BaseLoader(ABC):
    def load(self, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )


class SALLoader(BaseLoader):
    def __init__(self) -> None:
        from jet.data import sal

        self.sal = sal

    def load(self, shot_num: int, name: str, channels: Optional[list[str]] = None):
        try:
            return self.load_signal(shot_num, name)
        except Exception as e:
            raise MissingProfileError(f"{e}, {type(e)}")

    def load_signal(self, shot_num: int, name: str) -> xr.DataArray:
        signal = self.sal.get(f"/pulse/{shot_num}/ppf/signal/jetppf/{name}")

        attrs = {}
        attrs["units"] = signal.units
        attrs["description"] = signal.description

        coords = []
        for dim in signal.dimensions:
            coord = xr.DataArray(
                data=dim.data, attrs=dict(units=dim.units, description=dim.description)
            )
            coords.append(coord)

        data = xr.DataArray(data=signal.data, coords=coords, attrs=attrs)
        data.name = name
        data.to_dataset()
        return data


class UDALoader(BaseLoader):
    def __init__(self) -> None:
        pass

    def _get_client(self):
        import pyuda

        client = pyuda.Client()
        client.set_property("get_meta", True)
        client.set_property("timeout", 10)
        return client

    def load(self, shot_num: int, name: str, channels: Optional[list[str]] = None):
        try:
            if channels is not None:
                return self.load_channels(shot_num, name, channels)
            elif name in ["RBB", "RBA"]:
                return self.load_image(shot_num, name)
            else:
                return self.load_signal(shot_num, name)
        except Exception as e:
            raise MissingProfileError(f"{e}, {type(e)}")

    def load_channels(self, shot_num: int, name: str, channels: list[str]):
        signals = []
        loaded_channels = []
        for channel in channels:
            try:
                signal = self.load_signal(shot_num, channel)
                signals.append(signal)
                loaded_channels.append(channel)
            except MissingSourceError:
                continue

        if len(signals) == 0:
            raise MissingSourceError(
                f'Could not load profile {name} for shot "{shot_num}". Could not load any channels!'
            )

        channels = xr.DataArray(data=loaded_channels)
        channels.name = "channels"

        first_signal = signals[0]
        for i, signal in enumerate(signals):
            dim_map = {
                dim_name: new_dim_name
                for dim_name, new_dim_name in zip(signal.dims, first_signal.dims)
            }
            signal = signal.rename(dim_map)
            signal = signal.interp_like(first_signal, method="zero")
            signals[i] = signal

        signals = xr.concat(signals, dim=channels)
        return signals

    def load_signal(self, shot_num: int, name: str):
        import pyuda

        try:
            client = self._get_client()
            signal = client.get(name, shot_num)
            dataset = self._convert_signal_to_dataset(name, signal)
            dataset = dataset.squeeze(drop=True)
            return dataset
        except pyuda.ServerException as e:
            raise MissingSourceError(
                f'Could not load profile {name} for shot "{shot_num}". Encountered exception: {e}'
            )

    def _convert_signal_to_dataset(self, signal_name, signal):
        dim_names = self._normalize_dimension_names(signal)
        coords = {}
        for name, dim in zip(dim_names, signal.dims):
            data = dim.data

            coord = xr.DataArray(
                np.atleast_1d(data), dims=[name], attrs=dict(units=dim.units)
            )
            coords[name] = coord

        data = np.atleast_1d(signal.data)
        error = np.atleast_1d(signal.errors)
        attrs = self._get_dataset_attributes(signal_name, signal)

        data = xr.DataArray(data, dims=dim_names, coords=coords, attrs=attrs)
        data.name = signal_name

        error = xr.DataArray(error, dims=dim_names, coords=coords, attrs=attrs)
        error.name = f"{signal_name}_error"

        dataset = xr.Dataset({data.name: data, error.name: error})
        return dataset

    def load_image(self, shot_num: int, name: str) -> xr.Dataset:
        client = self._get_client()
        image = client.get_images(name, shot_num)
        dataset = self._convert_image_to_dataset(image)
        dataset.name = name
        dataset.attrs["shot_id"] = shot_num
        dataset = dataset.to_dataset()
        return dataset

    def _convert_image_to_dataset(self, image):
        attrs = {
            name: getattr(image, name)
            for name in dir(image)
            if not name.startswith("_") and not callable(getattr(image, name))
        }

        attrs.pop("frame_times")
        attrs.pop("frames")

        attrs["CLASS"] = "IMAGE"
        attrs["IMAGE_VERSION"] = "1.2"

        time = np.atleast_1d(image.frame_times)
        coords = {"time": xr.DataArray(time, dims=["time"], attrs=dict(units="s"))}

        if image.is_color:
            frames = [np.dstack((frame.r, frame.g, frame.b)) for frame in image.frames]
            frames = np.stack(frames)
            if frames.shape[1] != image.height:
                frames = np.swapaxes(frames, 1, 2)
            dim_names = ["time", "height", "width", "channel"]

            attrs["IMAGE_SUBCLASS"] = "IMAGE_TRUECOLOR"
            attrs["INTERLACE_MODE"] = "INTERLACE_PIXEL"
        else:
            frames = [frame.k for frame in image.frames]
            frames = np.stack(frames)
            frames = np.atleast_3d(frames)
            if frames.shape[1] != image.height:
                frames = np.swapaxes(frames, 1, 2)
            dim_names = ["time", "height", "width"]

            attrs["IMAGE_SUBCLASS"] = "IMAGE_INDEXED"

        dataset = xr.DataArray(frames, dims=dim_names, coords=coords, attrs=attrs)
        return dataset

    def _remove_exceptions(self, signal_name, signal):
        """Handles when signal attributes contain exception objects"""
        signal_attributes = dir(signal)
        for attribute in signal_attributes:
            try:
                getattr(signal, attribute)
            except UnicodeDecodeError as exception:
                print(f"{signal_name} {attribute}: {exception}")
                signal_attributes.remove(attribute)
        return signal_attributes

    def _get_signal_metadata_fields(self, signal, signal_name):
        """Retrieves the appropriate metadata field for a given signal"""
        return [
            attribute
            for attribute in self._remove_exceptions(signal_name, signal)
            if not attribute.startswith("_")
            and attribute not in ["data", "errors", "time", "meta", "dims"]
            and not callable(getattr(signal, attribute))
        ]

    def _get_dataset_attributes(self, signal_name: str, signal) -> dict:
        metadata = self._get_signal_metadata_fields(signal, signal_name)

        attrs = {}
        for field in metadata:
            try:
                attrs[field] = getattr(signal, field)
            except TypeError:
                pass

        for key, attr in attrs.items():
            if isinstance(attr, np.generic):
                attrs[key] = attr.item()
            elif isinstance(attr, np.ndarray):
                attrs[key] = attr.tolist()
            elif isinstance(attr, tuple):
                attrs[key] = list(attr)
            elif attr is None:
                attrs[key] = "null"

        return attrs

    def _normalize_dimension_names(self, signal):
        """Make the dimension names sensible"""
        dims = [dim.label for dim in signal.dims]
        count = 0
        dim_names = []
        empty_names = ["", " ", "-"]

        for name in dims:
            # Create names for unlabelled dims
            if name in empty_names:
                name = f"dim_{count}"
                count += 1

            # Normalize weird names to standard names
            dim_names.append(name)

        dim_names = list(map(lambda x: x.lower(), dim_names))
        dim_names = [re.sub("[^a-zA-Z0-9_\n\\.]", "", dim) for dim in dim_names]
        return dim_names


class ZarrLoader(BaseLoader):
    def __init__(self, base_path: str, **kwargs) -> None:
        self.base_path = base_path
        self.fs = fsspec.filesystem(**kwargs)

    def load(self, shot_num: int, name: str) -> xr.DataArray:
        source, name = name.split("/", maxsplit=1)
        url = f"{self.base_path}/{shot_num}.zarr/{source}"

        try:
            store = zarr.storage.FSStore(url, fs=self.fs)
            dataset = xr.open_zarr(store)
        except FileNotFoundError:
            raise MissingSourceError(
                f'Could not load profile {name} from "{url}". "{url}" does not exist.'
            )

        if name not in dataset:
            raise MissingProfileError(
                f'Could not load profile "{name}" from "{url}". "{name}" not in dataset.'
            )

        dataset = dataset[name]
        for name in ["format", "file_name"]:
            if name in dataset.attrs:
                dataset.attrs.pop(name)
        return dataset


class LoaderTypes(str, Enum):
    ZARR = "zarr"
    UDA = "uda"
    SAL = "sal"


loader_registry = Registry[BaseLoader]()
loader_registry.register(LoaderTypes.ZARR, ZarrLoader)
loader_registry.register(LoaderTypes.SAL, SALLoader)
loader_registry.register(LoaderTypes.UDA, UDALoader)
