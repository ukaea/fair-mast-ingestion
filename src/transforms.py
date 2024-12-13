import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pint
import pyarrow.parquet as pq
import xarray as xr

DIMENSION_MAPPING_FILE = "mappings/mast/dimensions.json"
UNITS_MAPPING_FILE = "mappings/mast/units.json"
CUSTOM_UNITS_FILE = "mappings/mast/custom_units.txt"


def get_dataset_item_uuid(name: str, shot: int) -> str:
    oid_name = name + "/" + str(shot)
    return str(uuid.uuid5(uuid.NAMESPACE_OID, oid_name))


class MapDict:
    def __init__(self, transform) -> None:
        self.transform = transform

    def __call__(self, datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
        out = {}
        for key, dataset in datasets.items():
            try:
                out[key] = self.transform(dataset)
            except Exception as e:
                raise RuntimeError(f"{key}: {e}")
        return out


class RenameDimensions:
    def __init__(
        self, mapping_file=DIMENSION_MAPPING_FILE, squeeze_dataset: bool = True
    ) -> None:
        self.squeeze_dataset = squeeze_dataset

        with Path(mapping_file).open("r") as handle:
            self.dimension_mapping = json.load(handle)

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        name = dataset.attrs["name"]
        if name in self.dimension_mapping:
            dims = self.dimension_mapping[name]

            for old_name, new_name in dims.items():
                if old_name in dataset.dims:
                    if new_name not in dataset:
                        dataset = dataset.rename_dims({old_name: new_name})
                    else:
                        dataset = dataset.swap_dims({old_name: new_name})
                        dataset = dataset.drop_vars(old_name)

            for old_name, new_name in dims.items():
                if old_name in dataset.coords:
                    dataset = dataset.rename_vars({old_name: new_name})

            dataset.attrs["dims"] = list(dataset.sizes.keys())
        if self.squeeze_dataset:
            dataset = dataset.squeeze()
        dataset = dataset.compute()
        return dataset


class DropZeroDimensions:
    def __call__(self, dataset: xr.Dataset) -> Any:
        for key, coord in dataset.coords.items():
            if (coord.values == 0).all():
                dataset = dataset.drop_vars(key)
        dataset = dataset.compute()
        return dataset


class DropZeroDataset:
    def __call__(self, dataset: xr.Dataset) -> Any:
        for key, item in dataset.data_vars.items():
            if (item.values == 0).all():
                dataset = dataset.drop_vars(key)
        dataset = dataset.compute()
        return dataset


class DropDatasets:
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
        for key in self.keys:
            datasets.pop(key)
        return datasets


class DropCoordinates:
    def __init__(self, name, keys: list[str]) -> None:
        self.name = name
        self.keys = keys

    def __call__(self, datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
        for name, dataset in datasets.items():
            if name == self.name:
                for key in self.keys:
                    dataset = dataset.drop_indexes(key)
                    datasets[name] = dataset.drop_vars(key)
        return datasets

    def _drop_unused_coords(self, data: xr.Dataset) -> xr.Dataset:
        used_coords = set()
        for var in data.data_vars.values():
            used_coords.update(var.dims)

        # Drop coordinates that are not used
        unused_coords = set(data.coords) - used_coords
        data = data.drop_vars(unused_coords)
        return data


class RenameVariables:
    def __init__(self, mapping: dict[str, str]):
        self.mapping = mapping

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        for key, value in self.mapping.items():
            if key in dataset:
                dataset = dataset.rename_vars({key: value})
        dataset = dataset.compute()
        return dataset


class MergeDatasets:
    def __call__(self, dataset_dict: dict[str, xr.Dataset]) -> xr.Dataset:
        dataset = xr.merge(dataset_dict.values())
        dataset = dataset.compute()
        dataset.attrs = {}
        return dataset


class TensoriseChannels:
    def __init__(
        self,
        stem: str,
        regex: Optional[str] = None,
        dim_name: Optional[str] = None,
        assign_coords: bool = True,
    ) -> None:
        self.stem = stem
        self.regex = regex if regex is not None else stem + "(\d+)"
        name = self.stem.split("/")[-1]
        self.dim_name = f"{name}_channel" if dim_name is None else dim_name
        self.assign_coords = assign_coords

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        group_keys = self._get_group_keys(dataset)

        # If we couldn't find any matching keys, do nothing.
        if len(group_keys) == 0:
            return dataset

        channels = [dataset[key] for key in group_keys]
        combined = xr.combine_nested(channels, concat_dim=self.dim_name)
        dataset[self.stem] = combined

        if self.assign_coords:
            dataset[self.stem] = dataset[self.stem].assign_coords(
                {self.dim_name: group_keys}
            )

        dataset[self.stem] = dataset[self.stem].chunk("auto")
        dataset[self.stem] = self._update_attributes(dataset[self.stem], channels)
        dataset = dataset.drop_vars(group_keys)
        dataset: xr.Dataset = dataset.compute()
        return dataset

    def _update_attributes(
        self, dataset: xr.Dataset, channels: list[xr.Dataset]
    ) -> xr.Dataset:
        attrs = channels[0].attrs
        channel_descriptions = [c.attrs.get("description", "") for c in channels]
        description = "\n".join(channel_descriptions)
        attrs["name"] = self.stem
        attrs["description"] = description
        attrs["channel_descriptions"] = channel_descriptions
        attrs["uuid"] = get_dataset_item_uuid(attrs["name"], attrs["shot_id"])
        attrs["shape"] = list(dataset.sizes.values())
        attrs["rank"] = len(attrs["shape"])
        attrs["dims"] = list(dataset.sizes.keys())
        attrs.pop("uda_name", "")
        attrs.pop("mds_name", "")
        dataset.attrs = attrs
        return dataset

    def _get_group_keys(self, dataset: xr.Dataset) -> list[str]:
        group_keys = dataset.data_vars.keys()
        group_keys = [
            key for key in group_keys if re.search(self.regex, key) is not None
        ]
        group_keys = self._sort_numerically(group_keys)
        return group_keys

    def _parse_digits(self, s):
        # Split the string into a list of numeric and non-numeric parts
        parts = re.split(self.regex, s)
        # Convert numeric parts to integers
        return [int(part) if part.isdigit() else part for part in parts]

    def _sort_numerically(self, strings: list[str]) -> list[str]:
        return sorted(strings, key=self._parse_digits)


class TransformUnits:
    def __init__(self):
        with Path(UNITS_MAPPING_FILE).open("r") as handle:
            self.units_map = json.load(handle)

        self.ureg = pint.UnitRegistry()
        self.ureg.load_definitions(CUSTOM_UNITS_FILE)

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        for array in dataset.data_vars.values():
            self._update_units(array)

        for array in dataset.coords.values():
            self._update_units(array)

        dataset = dataset.compute()
        return dataset

    def _update_units(self, array: xr.DataArray):
        units = array.attrs.get("units", "")
        units = self.units_map.get(units, units)
        units = self._parse_units(units)
        array.attrs["units"] = units

    def _parse_units(self, unit: str) -> str:
        try:
            unit = self.ureg.parse_units(unit)
            unit = format(unit, "~")
            return unit
        except Exception:
            return unit


class ASXTransform:
    """ASX is very special.

    The time points are actually the data and the data is blank.
    This transformation renames them and used the correct dimension mappings.
    """

    def __init__(self) -> None:
        with Path(DIMENSION_MAPPING_FILE).open("r") as handle:
            self.dimension_mapping = json.load(handle)

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = dataset.squeeze()
        name = dataset.attrs["name"]

        if name not in self.dimension_mapping:
            return dataset

        dataset = dataset.rename_dims(self.dimension_mapping[name])
        dataset = dataset.drop("data")
        dataset["data"] = dataset["time"]
        dataset = dataset.drop("time")
        dataset = dataset.compute()
        return dataset


class LCFSTransform:
    """LCFS transform for LCFS coordinates

    In MAST, the LCFS coordinates have a lot of padding.
    This transform groups the r and z parameters and crops the padding.
    """

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        if "lcfsr_c" not in dataset.data_vars:
            return dataset

        r = dataset["lcfsr_c"]
        fill_value = np.nanmax(r.values)
        max_index = np.max(np.argmax(r.values, axis=1))
        dataset = dataset.sel(lcfs_coords=dataset.lcfs_coords[:max_index])

        r = dataset["lcfsr_c"]
        z = dataset["lcfsz_c"]
        dataset["lcfsr_c"] = r.where(r.values != fill_value, np.nan)
        dataset["lcfsz_c"] = z.where(z.values != fill_value, np.nan)
        dataset = dataset.compute()
        return dataset


class AddGeometry:
    def __init__(self, stem: str, path: str):
        table = pq.read_table(path)
        geom_data = table.to_pandas()
        geom_data.drop("uda_name", inplace=True, axis=1)
        geom_data.columns = [stem + "_" + c for c in geom_data.columns]
        self.stem = stem
        index_name = f"{self.stem}_geometry_index"
        geom_data[index_name] = [
            f"{stem}{index+1:02}" for index in range(len(geom_data))
        ]
        geom_data = geom_data.set_index(index_name)
        self.geom_data = geom_data.to_xarray()

        if table.schema.metadata:
            arrow_metadata = {
                key.decode(): value.decode()
                for key, value in table.schema.metadata.items()
            }
            renamed_metadata = {"source": "geometry_source_file"}
            arrow_metadata = {
                renamed_metadata.get(key, key): value
                for key, value in arrow_metadata.items()
            }

        for field in table.schema:
            if field.metadata:
                field_metadata = {
                    key.decode(): value.decode()
                    for key, value in field.metadata.items()
                }
                self.geom_data[f"{stem}_{field.name}"].attrs.update(field_metadata)
                self.geom_data[f"{stem}_{field.name}"].attrs.update(arrow_metadata)

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        geom_data = self.geom_data.copy()
        dataset = xr.merge(
            [dataset, geom_data], combine_attrs="no_conflicts", join="left"
        )
        dataset = dataset.compute()
        return dataset


class AlignChannels:
    def __init__(self, source: str):
        self.source = source
        self.channel_dim = f"{source}_channel"
        self.geometry_dim = f"{source}_geometry_index"

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        geometry_index = dataset.coords[self.geometry_dim].values
        dataset = dataset.reindex({f"{self.source}_channel": geometry_index})
        dataset = dataset.drop_vars(self.geometry_dim)
        dataset = dataset.rename(
            {f"{self.source}_geometry_index": f"{self.source}_channel"}
        )

        return dataset


class XDCRenameDimensions:
    """XDC is a special boi...

    XDC has dynamically named time dimensions. The same signal can be called 'time2' or 'time4'
    depending on what got written to disk.
    """

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = dataset.squeeze()
        for dim_name in dataset.sizes.keys():
            if "time" in dim_name and dim_name != "time":
                dataset = dataset.rename_dims({dim_name: "time"})
                dataset = dataset.rename_vars({dim_name: "time"})

        dataset = dataset.compute()
        return dataset


class ProcessImage:
    def __call__(self, dataset: dict[str, xr.Dataset]) -> xr.Dataset:
        dataset: xr.Dataset = list(dataset.values())[0]
        dataset.attrs["units"] = "pixels"
        dataset.attrs["shape"] = list(dataset.sizes.values())
        dataset.attrs["rank"] = len(dataset.sizes.values())
        dataset = dataset.compute()
        return dataset


class ReplaceInvalidValues:
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = dataset.where(dataset != -999, np.nan)
        dataset = dataset.compute()
        return dataset
