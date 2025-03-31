from typing import Union

import numpy as np
import xarray as xr

from src.core.load import BaseLoader, MissingProfileError, MissingSourceError
from src.core.log import logger
from src.core.model import Dimension, Mapping, Source
from src.level2.transforms import (
    BackgroundSubtractionTransform,
    DatasetInterpolationTransform,
    transform_registry,
)


class DatasetReader:
    def __init__(self, mapping: Mapping, loader: BaseLoader) -> None:
        self._loader = loader
        self._mapping = mapping

    def set_shot(self, shot: int):
        self._shot = shot

    def read_dataset(self, shot: int, name: str) -> xr.Dataset:
        self.set_shot(shot)

        dataset = self.read_profiles(shot, name)

        if len(dataset) == 0:
            return dataset

        dataset = self.apply_interpolation(dataset, name)
        dataset = self.apply_transforms(dataset, name) #adding here?
        dataset = self.apply_attributes(dataset, name)
        return dataset

    def read_profiles(self, shot: int, dataset_name: str) -> dict[str, xr.DataArray]:
        self.set_shot(shot)

        dataset = self._mapping.datasets[dataset_name]

        profiles = {}
        for profile_name, profile_info in dataset.profiles.items():
            try:
                logger.debug(f"Create profile {profile_name}")
                profile = self.read_profile(shot, dataset_name, profile_name)
                logger.debug(f"Loaded profile {profile_name}")
            except MissingSourceError as e:
                logger.warning(e)
                continue
            except MissingProfileError as e:
                logger.warning(e)
                continue

            profiles[profile_name] = profile
        return profiles

    def read_profile(
        self, shot: int, dataset_name: str, profile_name: str
    ) -> xr.DataArray: #eg 30421 magnetics b_field_tor_probe_saddle_voltage
        self.set_shot(shot)

        profile = self._mapping.datasets[dataset_name].profiles[profile_name]
        source = self._get_source(profile.source)
        dataset = self._loader.load(self._shot, source.name, source.channels)
        coordinates = {}

        dim_names = list(profile.dimensions.keys())
        for dim_index, (dim_name, dim) in enumerate(profile.dimensions.items()):
            if dim is not None and dim.source is not None:
                # Get dimension from seperate source
                coordinates[dim_name] = self._create_dimension(dim_name, dim)
            elif dim_index < len(dataset.coords):
                # Get dimension from data array object
                names = list(dataset.sizes.keys())
                coord: xr.DataArray = dataset.coords[names[dim_index]]
                if coord.name != dim_name:
                    coord = coord.rename({coord.name: dim_name})
                coordinates[dim_name] = coord

            if dim is not None and dim.units is not None:
                coordinates[dim_name].attrs["units"] = dim.units

            if dim_name in coordinates:
                all_nan = (coordinates[dim_name] == np.nan).all()
                all_zero = (coordinates[dim_name] == 0).all()
                if all_nan or all_zero:
                    coordinates.pop(dim_name)

        dataset = dataset.squeeze()
        item = xr.DataArray(
            name=profile_name,
            data=dataset.values,
            dims=dim_names,
            coords=coordinates,
            attrs=dataset.attrs,
        )

        if profile.units is not None:
            item.attrs["units"] = profile.units

        if profile.imas is not None:
            item.attrs["imas"] = profile.imas

        item.attrs["description"] = profile.description
        item.attrs["name"] = profile_name

        item = item.where(np.isfinite(item.values), np.nan)
        if profile.fill_value is not None:
            item = item.where(item != profile.fill_value, np.nan)

        item *= profile.scale
        item = self._parse_units(item)
        item = self._convert_units(item, profile.target_units)
        item = item.sortby([name for name in dim_names if "channel" not in name])
        item = item.drop_duplicates(dim=...)

        if item.isnull().all():
            raise MissingProfileError(
                f"All values are NaN for shot {self._shot} and profile {profile_name}"
            )

        if source.background_window:
            start, end = source.background_window.tmin, source.background_window.tmax
            logger.info(f"Applying background subtraction for {profile_name} using window {start}-{end}")
            subtractor = BackgroundSubtractionTransform(start, end)
            print(subtractor)
            item = subtractor.transform_array(item)
        return item

    def apply_interpolation(self, dataset: xr.Dataset, dataset_name: str) -> xr.Dataset:
        dataset_config = self._mapping.datasets[dataset_name]
        global_params = self._mapping.global_interpolate
        interpolator = DatasetInterpolationTransform(dataset_config, global_params)
        dataset = interpolator.transform(dataset)
        return dataset

    def apply_transforms(self, dataset: xr.Dataset, dataset_name: str) -> xr.Dataset:
        dataset_config = self._mapping.datasets[dataset_name]
        if dataset_config.transforms is None:
            return dataset

        for name, params in dataset_config.transforms.items():
            params = {} if params is None else params
            transformer = transform_registry.create(name, **params)
            dataset = transformer.transform(dataset)

        return dataset

    def apply_attributes(
        self,
        dataset: xr.Dataset,
        name: str,
    ) -> xr.Dataset:
        dataset.attrs["name"] = name
        dataset.attrs["description"] = self._mapping.datasets[name].description
        dataset.attrs["imas"] = self._mapping.datasets[name].imas
        return dataset

    def _parse_units(self, item: xr.DataArray):
        if "units" not in item.attrs:
            logger.warning(f'{self._shot} and signal "{item.name}" has no units')
            return item

        try:
            item = item.pint.quantify()
            item = item.pint.dequantify(format="#~")
        except TypeError as e:
            logger.warning(
                f'Issue with parsing units for {self._shot} and signal "{item.name}": {e}'
            )
        except ValueError as e:
            logger.warning(
                f'Issue with parsing units for {self._shot} and signal "{item.name}": {e}'
            )
        except AssertionError as e:
            logger.warning(
                f'Issue with parsing units for {self._shot} and signal "{item.name}": {e}'
            )

        return item

    def _convert_units(self, item: xr.DataArray, target_units: str):
        if target_units is None:
            return item

        if "units" not in item.attrs:
            raise RuntimeError(
                f"Cannot convert to target units {target_units} for shot {self.shot} and signal {item.name}. No units defined."
            )

        try:
            item = item.pint.quantify()
            item = item.pint.to(target_units)
            item = item.pint.dequantify(format="#~")
        except ValueError as e:
            logger.warning(
                f'Issue with converting units for {self._shot} and signal "{item.name}": {e}'
            )
        except AssertionError as e:
            logger.warning(
                f'Issue with converting units for {self._shot} and signal "{item.name}": {e}'
            )

        return item

    def _create_dimension(self, dim_name: str, dimension: Dimension) -> xr.DataArray:
        dim_source = self._get_source(dimension.source)
        data = self._loader.load(self._shot, dim_source.name)
        data = data.squeeze()

        attrs = {}
        for name in ["units", "description", "label"]:
            attrs[name] = data.attrs[name]

        data = data.values
        ndims = len(data.shape)
        if ndims == 1:
            coord = xr.DataArray(data=data, dims=[dim_name], attrs=attrs)
        elif ndims > 1:
            coord = xr.DataArray(data=data[0], dims=[dim_name], attrs=attrs)

        coord.name = dim_name
        coord *= dimension.scale

        if dimension.units is not None:
            coord.attrs["units"] = dimension.units
        if dimension.imas is not None:
            coord.attrs["imas"] = dimension.imas

        coord = self._parse_units(coord)
        coord = self._convert_units(coord, dimension.target_units)

        return coord

    def _get_source(self, sources: Union[Source, str]) -> Source:
        if isinstance(sources, list):
            if len(sources) == 1:
                return sources[0]

            for source in sources:
                shot_range = source.shot_range
                if shot_range.shot_min <= self._shot < shot_range.shot_max:
                    return source

            raise MissingProfileError(
                f"Cannot find valid source for profile for shot {self._shot}"
            )
        elif isinstance(sources, str):
            source = Source(name=sources)
        else:
            raise RuntimeError(f"Unknown source type {type(sources)}")

        return source
