import json
import re
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
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
                    dataset = dataset.rename_dims({old_name: new_name})

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


class StandardiseSignalDataset:
    def __init__(self, source: str, squeeze_dataset: bool = True) -> None:
        self.source = source
        self.squeeze_dataset = squeeze_dataset

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.squeeze_dataset:
            dataset = dataset.squeeze(drop=True)

        name = dataset.attrs["name"].split("/")[-1]

        # Drop error if all zeros
        if (dataset["error"].values == 0).all():
            dataset = dataset.drop_vars("error")

        # Rename variables
        new_names = {}
        if "error" in dataset:
            new_names["data"] = name
            new_names["error"] = "_".join([name, "error"])
        else:
            name = (
                name + "_"
                if name == "time" or name in dataset.data_vars or name in dataset.coords
                else name
            )
            new_names["data"] = name

        dataset = dataset.rename(new_names)
        dataset = self._drop_unused_coords(dataset)

        if "time" in dataset.dims:
            dataset = dataset.drop_duplicates(dim="time")

        # Update attributes
        attrs = dataset.attrs
        attrs["name"] = self.source + "/" + new_names["data"]
        attrs["dims"] = list(dataset.sizes.keys())
        dataset[new_names["data"]].attrs = attrs
        dataset = dataset.compute()
        return dataset

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


class AddXSXCameraParams:
    def __init__(self, stem: str, path: str):
        cam_data = pd.read_csv(path)
        cam_data.drop("name", inplace=True, axis=1)
        cam_data.drop("comment", inplace=True, axis=1)
        cam_data.columns = [stem + "_" + c for c in cam_data.columns]
        self.stem = stem
        index_name = f"{self.stem}_channel"
        cam_data[index_name] = [
            stem + "_" + str(index + 1) for index in range(len(cam_data))
        ]
        cam_data = cam_data.set_index(index_name)
        self.cam_data = cam_data.to_xarray()

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        cam_data = self.cam_data.copy()
        # if camera data in not in dataset, then skip and do nothing
        if self.stem not in dataset:
            return dataset
        dataset = xr.merge(
            [dataset, cam_data], combine_attrs="drop_conflicts", join="left"
        )
        dataset = dataset.compute()
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


class Pipeline:
    def __init__(self, transforms: list):
        self.transforms = transforms

    def __call__(self, x: Any) -> Any:
        for transform in self.transforms:
            x = transform(x)
        return x


class PipelineRegistry:
    def __init__(self) -> None:
        pass

    def get(self, name: str) -> Pipeline:
        if name not in self.pipelines:
            raise RuntimeError(f"{name} is not a registered source!")
        return self.pipelines[name]


class ReplaceInvalidValues:
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = dataset.where(dataset != -999, np.nan)
        dataset = dataset.compute()
        return dataset


class MASTUPipelineRegistry(PipelineRegistry):
    def __init__(self) -> None:
        dim_mapping_file = "mappings/mastu/dimensions.json"

        self.pipelines = {
            "amb": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("amb")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amc": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("amc")),
                    MergeDatasets(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "ip": "plasma_current",
                        }
                    ),
                ]
            ),
            "anb": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("anb")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "act": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("act")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "acu": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("anb")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("ayc")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ayd": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("ayd")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "epm": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("epm")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "esm": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("esm")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xsx": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("xsx")),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_ch(\d+)"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_ch(\d+)"),
                    TensoriseChannels("tcam", regex=r"tcam_ch(\d+)"),
                ]
            ),
            "xdc": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("xdc")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
        }


class MASTPipelineRegistry(PipelineRegistry):
    def __init__(self) -> None:
        self.pipelines = {
            "abm": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(DropZeroDimensions()),
                    MapDict(StandardiseSignalDataset("abm")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "acc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("acc")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "act": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("act")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ada": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ada")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aga": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("aga")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "adg": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("adg")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ahx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ahx")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aim": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("aim")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "air": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("air")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ait": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ait")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "alp": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(DropZeroDimensions()),
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("alp")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ama": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ama")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amb": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("amb")),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("ccbv"),
                    AddGeometry("ccbv", "geometry/data/amb/ccbv.parquet"),
                    AlignChannels("ccbv"),
                    TensoriseChannels("fl_cc"),
                    AddGeometry("fl_cc", "geometry/data/amb/fl_cc.parquet"),
                    AlignChannels("fl_cc"),
                    TensoriseChannels("fl_p2l", regex=r"fl_p2l_(\d+)"),
                    AddGeometry("fl_p2l", "geometry/data/amb/fl_p2l.parquet"),
                    AlignChannels("fl_p2l"),
                    TensoriseChannels("fl_p3l", regex=r"fl_p3l_(\d+)"),
                    AddGeometry("fl_p3l", "geometry/data/amb/fl_p3l.parquet"),
                    AlignChannels("fl_p3l"),
                    TensoriseChannels("fl_p4l", regex=r"fl_p4l_(\d+)"),
                    AddGeometry("fl_p4l", "geometry/data/amb/fl_p4l.parquet"),
                    AlignChannels("fl_p4l"),
                    TensoriseChannels("fl_p5l", regex=r"fl_p5l_(\d+)"),
                    AddGeometry("fl_p5l", "geometry/data/amb/fl_p5l.parquet"),
                    AlignChannels("fl_p5l"),
                    TensoriseChannels("fl_p6l", regex=r"fl_p6l_(\d+)"),
                    AddGeometry("fl_p6l", "geometry/data/amb/fl_p6l.parquet"),
                    AlignChannels("fl_p6l"),
                    TensoriseChannels("fl_p2u", regex=r"fl_p2u_(\d+)"),
                    AddGeometry("fl_p2u", "geometry/data/amb/fl_p2u.parquet"),
                    AlignChannels("fl_p2u"),
                    TensoriseChannels("fl_p3u", regex=r"fl_p3u_(\d+)"),
                    AddGeometry("fl_p3u", "geometry/data/amb/fl_p3u.parquet"),
                    AlignChannels("fl_p3u"),
                    TensoriseChannels("fl_p4u", regex=r"fl_p4u_(\d+)"),
                    AddGeometry("fl_p4u", "geometry/data/amb/fl_p4u.parquet"),
                    AlignChannels("fl_p4u"),
                    TensoriseChannels("fl_p5u", regex=r"fl_p5u_(\d+)"),
                    AddGeometry("fl_p5u", "geometry/data/amb/fl_p5u.parquet"),
                    AlignChannels("fl_p5u"),
                    TensoriseChannels("obr"),
                    AddGeometry("obr", "geometry/data/amb/xma_obr.parquet"),
                    AlignChannels("obr"),
                    TensoriseChannels("obv"),
                    AddGeometry("obv", "geometry/data/amb/xma_obv.parquet"),
                    AlignChannels("obv"),
                ]
            ),
            "amc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("amc")),
                    MergeDatasets(),
                    TransformUnits(),
                    AddGeometry(
                        "p2il_coil_current",
                        "geometry/data/amc/amc_p2il_coil_current.parquet",
                    ),
                    AlignChannels("p2il_coil_current"),
                    AddGeometry(
                        "p2iu_coil_current",
                        "geometry/data/amc/amc_p2iu_coil_current.parquet",
                    ),
                    AlignChannels("p2iu_coil_current"),
                    AddGeometry(
                        "p2l_case_current",
                        "geometry/data/amc/amc_p2l_case_current.parquet",
                    ),
                    AlignChannels("p2l_case_current"),
                    AddGeometry(
                        "p2ol_coil_current",
                        "geometry/data/amc/amc_p2ol_coil_current.parquet",
                    ),
                    AlignChannels("p2ol_coil_current"),
                    AddGeometry(
                        "p2ou_coil_current",
                        "geometry/data/amc/amc_p2ou_coil_current.parquet",
                    ),
                    AlignChannels("p2ou_coil_current"),
                    AddGeometry(
                        "p2u_case_current",
                        "geometry/data/amc/amc_p2u_case_current.parquet",
                    ),
                    AlignChannels("p2u_case_current"),
                    AddGeometry(
                        "p3l_case_current",
                        "geometry/data/amc/amc_p3l_case_current.parquet",
                    ),
                    AlignChannels("p3l_case_current"),
                    AddGeometry(
                        "p3l_coil_current",
                        "geometry/data/amc/amc_p3l_coil_current.parquet",
                    ),
                    AlignChannels("p3l_coil_current"),
                    AddGeometry(
                        "p3u_case_current",
                        "geometry/data/amc/amc_p3u_case_current.parquet",
                    ),
                    AlignChannels("p3u_case_current"),
                    AddGeometry(
                        "p3u_coil_current",
                        "geometry/data/amc/amc_p3u_coil_current.parquet",
                    ),
                    AlignChannels("p3u_coil_current"),
                    AddGeometry(
                        "p4l_case_current",
                        "geometry/data/amc/amc_p4l_case_current.parquet",
                    ),
                    AlignChannels("p4l_case_current"),
                    AddGeometry(
                        "p4l_coil_current",
                        "geometry/data/amc/amc_p4l_coil_current.parquet",
                    ),
                    AlignChannels("p4l_coil_current"),
                    AddGeometry(
                        "p4u_case_current",
                        "geometry/data/amc/amc_p4u_case_current.parquet",
                    ),
                    AlignChannels("p4u_case_current"),
                    AddGeometry(
                        "p4u_coil_current",
                        "geometry/data/amc/amc_p4u_coil_current.parquet",
                    ),
                    AlignChannels("p4u_coil_current"),
                    AddGeometry(
                        "p5l_case_current",
                        "geometry/data/amc/amc_p5l_case_current.parquet",
                    ),
                    AlignChannels("p5l_case_current"),
                    AddGeometry(
                        "p5l_coil_current",
                        "geometry/data/amc/amc_p5l_coil_current.parquet",
                    ),
                    AlignChannels("p5l_coil_current"),
                    AddGeometry(
                        "p5u_case_current",
                        "geometry/data/amc/amc_p5u_case_current.parquet",
                    ),
                    AlignChannels("p5u_case_current"),
                    AddGeometry(
                        "p5u_coil_current",
                        "geometry/data/amc/amc_p5u_coil_current.parquet",
                    ),
                    AlignChannels("p5u_coil_current"),
                    AddGeometry(
                        "p6l_case_current",
                        "geometry/data/amc/amc_p6l_case_current.parquet",
                    ),
                    AlignChannels("p6l_case_current"),
                    AddGeometry(
                        "p6l_coil_current",
                        "geometry/data/amc/amc_p6l_coil_current.parquet",
                    ),
                    AlignChannels("p6l_coil_current"),
                    AddGeometry(
                        "p6u_case_current",
                        "geometry/data/amc/amc_p6u_case_current.parquet",
                    ),
                    AlignChannels("p6u_case_current"),
                    AddGeometry(
                        "p6u_coil_current",
                        "geometry/data/amc/amc_p6u_coil_current.parquet",
                    ),
                    AlignChannels("p6u_coil_current"),
                    AddGeometry(
                        "sol_current", "geometry/data/amc/amc_sol_current.parquet"
                    ),
                    AlignChannels("sol_current"),
                ]
            ),
            "amh": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("amh")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "amm": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("amm")),
                    MergeDatasets(),
                    TransformUnits(),
                    AddGeometry("botcol", "geometry/data/amm/amm_botcol.parquet"),
                    AlignChannels("botcol"),
                    AddGeometry(
                        "endcrown_l", "geometry/data/amm/amm_endcrown_l.parquet"
                    ),
                    AlignChannels("endcrown_l"),
                    AddGeometry(
                        "endcrown_u", "geometry/data/amm/amm_endcrown_u.parquet"
                    ),
                    AlignChannels("endcrown_u"),
                    TensoriseChannels("incon"),
                    AddGeometry("incon", "geometry/data/amm/amm_incon.parquet"),
                    AlignChannels("incon"),
                    TensoriseChannels("lhorw"),
                    AddGeometry("lhorw", "geometry/data/amm/amm_lhorw.parquet"),
                    AlignChannels("lhorw"),
                    TensoriseChannels("mid"),
                    AddGeometry("mid", "geometry/data/amm/amm_mid.parquet"),
                    AlignChannels("mid"),
                    AddGeometry("p2larm1", "geometry/data/amm/amm_p2larm1.parquet"),
                    AlignChannels("p2larm1"),
                    AddGeometry("p2larm2", "geometry/data/amm/amm_p2larm2.parquet"),
                    AlignChannels("p2larm2"),
                    AddGeometry("p2larm3", "geometry/data/amm/amm_p2larm3.parquet"),
                    AlignChannels("p2larm3"),
                    AddGeometry("p2ldivpl1", "geometry/data/amm/amm_p2ldivpl1.parquet"),
                    AlignChannels("p2ldivpl1"),
                    AddGeometry("p2ldivpl2", "geometry/data/amm/amm_p2ldivpl2.parquet"),
                    AlignChannels("p2ldivpl2"),
                    AddGeometry("p2uarm1", "geometry/data/amm/amm_p2uarm1.parquet"),
                    AlignChannels("p2uarm1"),
                    AddGeometry("p2uarm2", "geometry/data/amm/amm_p2uarm2.parquet"),
                    AlignChannels("p2uarm2"),
                    AddGeometry("p2uarm3", "geometry/data/amm/amm_p2uarm3.parquet"),
                    AlignChannels("p2uarm3"),
                    AddGeometry("p2udivpl1", "geometry/data/amm/amm_p2udivpl1.parquet"),
                    AlignChannels("p2udivpl1"),
                    TensoriseChannels("ring"),
                    AddGeometry("ring", "geometry/data/amm/amm_ring.parquet"),
                    AlignChannels("ring"),
                    TensoriseChannels("rodgr"),
                    AddGeometry("rodgr", "geometry/data/amm/amm_rodr.parquet"),
                    AlignChannels("rodgr"),
                    AddGeometry("topcol", "geometry/data/amm/amm_topcol.parquet"),
                    AlignChannels("topcol"),
                    TensoriseChannels("uhorw"),
                    AddGeometry("uhorw", "geometry/data/amm/amm_uhorw.parquet"),
                    AlignChannels("uhorw"),
                    TensoriseChannels("vertw"),
                    AddGeometry("vertw", "geometry/data/amm/amm_vertw.parquet"),
                    AlignChannels("vertw"),
                ]
            ),
            "ams": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ams")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anb": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("amb")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ane": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ane")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "ant": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ant")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "anu": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("anu")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "aoe": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("aoe")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "arp": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("arp")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asb": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("asb")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "asm": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("asm")),
                    MergeDatasets(),
                    TensoriseChannels("sad_m"),
                    TransformUnits(),
                ]
            ),
            "asx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(ASXTransform()),
                    MapDict(StandardiseSignalDataset("asx")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "atm": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("atm")),
                    MergeDatasets(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "r": "radius",
                        }
                    ),
                ]
            ),
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ayc")),
                    DropCoordinates("ayc/segment_number", ["time_segment"]),
                    DropDatasets(["ayc/time"]),
                    MergeDatasets(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "r": "radius",
                        }
                    ),
                ]
            ),
            "aye": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("aye")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "efm": Pipeline(
                [
                    DropDatasets(
                        [
                            "efm/fcoil_n",
                            "efm/fcoil_segs_n",
                            "efm/limitern",
                            "efm/magpr_n",
                            "efm/silop_n",
                            "efm/shot_number",
                        ]
                    ),
                    MapDict(ReplaceInvalidValues()),
                    MapDict(DropZeroDimensions()),
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("efm")),
                    MergeDatasets(),
                    LCFSTransform(),
                    TransformUnits(),
                    RenameVariables(
                        {
                            "plasma_currc": "plasma_current_c",
                            "plasma_currx": "plasma_current_x",
                            "plasma_currrz": "plasma_current_rz",
                            "lcfsr_c": "lcfs_r",
                            "lcfsz_c": "lcfs_z",
                        }
                    ),
                ]
            ),
            "esm": Pipeline(
                [
                    MapDict(DropZeroDimensions()),
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("esm")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "esx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("esx")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "rba": Pipeline([ProcessImage()]),
            "rbb": Pipeline([ProcessImage()]),
            "rbc": Pipeline([ProcessImage()]),
            "rcc": Pipeline([ProcessImage()]),
            "rca": Pipeline([ProcessImage()]),
            "rco": Pipeline([ProcessImage()]),
            "rdd": Pipeline([ProcessImage()]),
            "rgb": Pipeline([ProcessImage()]),
            "rgc": Pipeline([ProcessImage()]),
            "rir": Pipeline([ProcessImage()]),
            "rit": Pipeline([ProcessImage()]),
            "xdc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xdc")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xim": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xim")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xmo": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xmo")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xpc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xpc")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xsx": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xsx")),
                    MergeDatasets(),
                    RenameVariables(
                        {
                            "hcaml#1": "hcam_l_1",
                            "hcaml#10": "hcam_l_10",
                            "hcaml#11": "hcam_l_11",
                            "hcaml#12": "hcam_l_12",
                            "hcaml#13": "hcam_l_13",
                            "hcaml#14": "hcam_l_14",
                            "hcaml#15": "hcam_l_15",
                            "hcaml#16": "hcam_l_16",
                            "hcaml#17": "hcam_l_17",
                            "hcaml#18": "hcam_l_18",
                            "hcaml#2": "hcam_l_2",
                            "hcaml#3": "hcam_l_3",
                            "hcaml#4": "hcam_l_4",
                            "hcaml#5": "hcam_l_5",
                            "hcaml#6": "hcam_l_6",
                            "hcaml#7": "hcam_l_7",
                            "hcaml#8": "hcam_l_8",
                            "hcaml#9": "hcam_l_9",
                            "hcamu#1": "hcam_u_1",
                            "hcamu#10": "hcam_u_10",
                            "hcamu#11": "hcam_u_11",
                            "hcamu#12": "hcam_u_12",
                            "hcamu#13": "hcam_u_13",
                            "hcamu#14": "hcam_u_14",
                            "hcamu#15": "hcam_u_15",
                            "hcamu#16": "hcam_u_16",
                            "hcamu#17": "hcam_u_17",
                            "hcamu#18": "hcam_u_18",
                            "hcamu#2": "hcam_u_2",
                            "hcamu#3": "hcam_u_3",
                            "hcamu#4": "hcam_u_4",
                            "hcamu#5": "hcam_u_5",
                            "hcamu#6": "hcam_u_6",
                            "hcamu#7": "hcam_u_7",
                            "hcamu#8": "hcam_u_8",
                            "hcamu#9": "hcam_u_9",
                            "tcam#1": "tcam_1",
                            "tcam#10": "tcam_10",
                            "tcam#11": "tcam_11",
                            "tcam#12": "tcam_12",
                            "tcam#13": "tcam_13",
                            "tcam#14": "tcam_14",
                            "tcam#15": "tcam_15",
                            "tcam#16": "tcam_16",
                            "tcam#17": "tcam_17",
                            "tcam#18": "tcam_18",
                            "tcam#2": "tcam_2",
                            "tcam#3": "tcam_3",
                            "tcam#4": "tcam_4",
                            "tcam#5": "tcam_5",
                            "tcam#6": "tcam_6",
                            "tcam#7": "tcam_7",
                            "tcam#8": "tcam_8",
                            "tcam#9": "tcam_9",
                        }
                    ),
                    TransformUnits(),
                    TensoriseChannels("v_ste29", regex=r"v_ste29_(\d+)"),
                    AddGeometry(
                        "v_ste29", "geometry/data/xsx/ssx_inner_vertical_cam.parquet"
                    ),
                    AlignChannels("v_ste29"),
                    TensoriseChannels("hcam_l", regex=r"hcam_l_(\d+)"),
                    AddGeometry(
                        "hcam_l", "geometry/data/xsx/ssx_lower_horizontal_cam.parquet"
                    ),
                    AlignChannels("hcam_l"),
                    TensoriseChannels("tcam", regex=r"tcam_(\d+)"),
                    AddGeometry("tcam", "geometry/data/xsx/ssx_tangential_cam.parquet"),
                    AlignChannels("tcam"),
                    TensoriseChannels("hpzr", regex=r"hpzr_(\d+)"),
                    AddGeometry(
                        "hpzr", "geometry/data/xsx/ssx_third_horizontal_cam.parquet"
                    ),
                    AlignChannels("hpzr"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_(\d+)"),
                    AddGeometry(
                        "hcam_u", "geometry/data/xsx/ssx_upper_horizontal_cam.parquet"
                    ),
                    AlignChannels("hcam_u"),
                    TensoriseChannels("v_ste36", regex=r"v_ste36_(\d+)"),
                    AddGeometry(
                        "v_ste36", "geometry/data/xsx/ssx_outer_vertical_cam.parquet"
                    ),
                    AlignChannels("v_ste36"),
                ]
            ),
            "xma": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xma")),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("ccbv", regex=r"ccbv_(\d+)"),
                    AddGeometry("ccbv", "geometry/data/xma/ccbv.parquet"),
                    AlignChannels("ccbv"),
                    TensoriseChannels("fl_cc"),
                    AddGeometry("fl_cc", "geometry/data/xma/fl_cc.parquet"),
                    AlignChannels("fl_cc"),
                    TensoriseChannels("fl_p2l"),
                    AddGeometry("fl_p2l", "geometry/data/xma/fl_p2l.parquet"),
                    AlignChannels("fl_p2l"),
                    TensoriseChannels("fl_p3l"),
                    AddGeometry("fl_p3l", "geometry/data/xma/fl_p3l.parquet"),
                    AlignChannels("fl_p3l"),
                    TensoriseChannels("fl_p4l"),
                    AddGeometry("fl_p4l", "geometry/data/xma/fl_p4l.parquet"),
                    AlignChannels("fl_p4l"),
                    TensoriseChannels("fl_p5l"),
                    AddGeometry("fl_p5l", "geometry/data/xma/fl_p5l.parquet"),
                    AlignChannels("fl_p5l"),
                    TensoriseChannels("fl_p6l"),
                    AddGeometry("fl_p6l", "geometry/data/xma/fl_p6l.parquet"),
                    AlignChannels("fl_p6l"),
                    TensoriseChannels("fl_p2u"),
                    AddGeometry("fl_p2u", "geometry/data/xma/fl_p2u.parquet"),
                    AlignChannels("fl_p2u"),
                    TensoriseChannels("fl_p3u"),
                    AddGeometry("fl_p3u", "geometry/data/xma/fl_p3u.parquet"),
                    AlignChannels("fl_p3u"),
                    TensoriseChannels("fl_p4u"),
                    AddGeometry("fl_p4u", "geometry/data/xma/fl_p4u.parquet"),
                    AlignChannels("fl_p4u"),
                    TensoriseChannels("fl_p5u"),
                    AddGeometry("fl_p5u", "geometry/data/xma/fl_p5u.parquet"),
                    AlignChannels("fl_p5u"),
                    TensoriseChannels("obr", regex=r"obr_(\d+)"),
                    AddGeometry("obr", "geometry/data/xma/xma_obr.parquet"),
                    AlignChannels("obr"),
                    TensoriseChannels("obv", regex=r"obv_(\d+)"),
                    AddGeometry("obv", "geometry/data/xma/xma_obv.parquet"),
                    AlignChannels("obv"),
                ]
            ),
            "xmb": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xmb")),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("sad_out_l"),
                    AddGeometry("sad_out_l", "geometry/data/xmb/xmb_sad_l.parquet"),
                    AlignChannels("sad_out_l"),
                    TensoriseChannels("sad_out_u"),
                    AddGeometry("sad_out_u", "geometry/data/xmb/xmb_sad_u.parquet"),
                    AlignChannels("sad_out_u"),
                    TensoriseChannels("sad_out_m"),
                    AddGeometry("sad_out_m", "geometry/data/xmb/xmb_sad_m.parquet"),
                    AlignChannels("sad_out_m"),
                ]
            ),
            "xmc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xmc")),
                    MergeDatasets(),
                    TransformUnits(),
                    TensoriseChannels("cc_mt", regex=r"cc_mt_(\d+)"),
                    AddGeometry("cc_mt", "geometry/data/xmc/ccmt.parquet"),
                    AlignChannels("cc_mt"),
                    TensoriseChannels("cc_mv", regex=r"cc_mv_(\d+)"),
                    AddGeometry("cc_mv", "geometry/data/xmc/ccmv.parquet"),
                    AlignChannels("cc_mv"),
                    TensoriseChannels("omv", regex=r"omv_(\d+)"),
                    AddGeometry("omv", "geometry/data/xmc/xmc_omv.parquet"),
                    AlignChannels("omv"),
                ]
            ),
            "xmp": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xmp")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "xms": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("xms")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
        }
