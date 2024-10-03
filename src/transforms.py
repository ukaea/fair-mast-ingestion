from typing import Any, Optional
import pint
import re
import json
import uuid
import numpy as np
import pandas as pd
import xarray as xr
from pathlib import Path

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

    def __init__(self, mapping_file = DIMENSION_MAPPING_FILE) -> None:
        with Path(mapping_file).open("r") as handle:
            self.dimension_mapping = json.load(handle)

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        name = dataset.attrs["name"]
        if name in self.dimension_mapping:
            dims = self.dimension_mapping[name]
            dataset = dataset.rename_dims(self.dimension_mapping[name])
            for old_name, new_name in dims.items():
                if old_name in dataset.coords:
                    dataset = dataset.rename_vars({old_name: new_name})
            dataset.attrs["dims"] = list(dataset.sizes.keys())
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


class StandardiseSignalDataset:

    def __init__(self, source: str) -> None:
        self.source = source

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
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
            name = name + "_" if name == "time" or name in dataset.data_vars or name in dataset.coords else name
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


class AddXSXCameraParams:

    def __init__(self, stem: str, path: str):
        cam_data = pd.read_csv(path)
        cam_data.drop("name", inplace=True, axis=1)
        cam_data.drop("comment", inplace=True, axis=1)
        cam_data.columns = [stem + "_" + c for c in cam_data.columns]
        self.stem = stem
        index_name = f'{self.stem}_channel'
        cam_data[index_name] = [stem + '_' + str(index+1) for index in range(len(cam_data))]
        cam_data = cam_data.set_index(index_name)
        self.cam_data = cam_data.to_xarray()

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        cam_data = self.cam_data.copy()
        # if camera data in not in dataset, then skip and do nothing
        if self.stem not in dataset:
            return dataset
        dataset = xr.merge([dataset, cam_data], combine_attrs="drop_conflicts", join='left')
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


class MASTUPipelineRegistry(PipelineRegistry):

    def __init__(self) -> None:
        dim_mapping_file = "mappings/mastu/dimensions.json"

        self.pipelines = {
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("ayc")),
                    MergeDatasets(),
                    TransformUnits(),
                ]
            ),
            "epm": Pipeline(
                [
                    # DropDatasets(
                    #     [
                    #         "efm/fcoil_n",
                    #         "efm/fcoil_segs_n",
                    #         "efm/limitern",
                    #         "efm/magpr_n",
                    #         "efm/silop_n",
                    #         "efm/shot_number",
                    #     ]
                    # ),
                    # MapDict(DropZeroDimensions()),
                    MapDict(RenameDimensions(dim_mapping_file)),
                    MapDict(StandardiseSignalDataset("epm")),
                    MergeDatasets(),
                    # LCFSTransform(),
                    TransformUnits(),
                    # RenameVariables(
                    #     {
                    #         "plasma_currc": "plasma_current_c",
                    #         "plasma_currx": "plasma_current_x",
                    #         "plasma_currrz": "plasma_current_rz",
                    #         "lcfsr_c": "lcfs_r",
                    #         "lcfsz_c": "lcfs_z",
                    #     }
                    # )
                ]
            )
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
                    MapDict(StandardiseSignalDataset("abm")),
                    MergeDatasets(),
                    TensoriseChannels("ccbv"),
                    TensoriseChannels("obr"),
                    TensoriseChannels("obv"),
                    TensoriseChannels("fl_cc"),
                    TensoriseChannels("fl_p"),
                    TransformUnits(),
                ]
            ),
            "amc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("amc")),
                    MergeDatasets(),
                    TransformUnits(),
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
                    TensoriseChannels("incon"),
                    TensoriseChannels("mid"),
                    TensoriseChannels("ring"),
                    TensoriseChannels("rodgr"),
                    TensoriseChannels("vertw"),
                    TensoriseChannels("lhorw"),
                    TensoriseChannels("uhorw"),
                    TransformUnits(),
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
            "ayc": Pipeline(
                [
                    MapDict(RenameDimensions()),
                    MapDict(StandardiseSignalDataset("ayc")),
                    MergeDatasets(),
                    TransformUnits(),
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
            "rgb": Pipeline([ProcessImage()]),
            "rgc": Pipeline([ProcessImage()]),
            "rir": Pipeline([ProcessImage()]),
            "rit": Pipeline([ProcessImage()]),
            "xdc": Pipeline(
                [

                    # MapDict(RenameVariables({
                    #     "xdc1/botcol": "ai_botcol",
                    #     "xdc1/camera_ok": "ai_camera_ok",
                    #     "xdc1/cam_ok": "ai_cam_ok",
                    #     "xdc1/ccbv01": "ai_ccbv01",
                    #     "xdc1/ccbv02": "ai_ccbv02",
                    #     "xdc1/ccbv03": "ai_ccbv03",
                    #     "xdc1/ccbv04": "ai_ccbv04",
                    #     "xdc1/ccbv05": "ai_ccbv05",
                    #     "xdc1/ccbv06": "ai_ccbv06",
                    #     "xdc1/ccbv07": "ai_ccbv07",
                    #     "xdc1/ccbv08": "ai_ccbv08",
                    #     "xdc1/ccbv09": "ai_ccbv09",
                    #     "xdc1/ccbv10": "ai_ccbv10",
                    #     "xdc1/ccbv11": "ai_ccbv11",
                    #     "xdc1/ccbv12": "ai_ccbv12",
                    #     "xdc1/ccbv13": "ai_ccbv13",
                    #     "xdc1/ccbv14": "ai_ccbv14",
                    #     "xdc1/ccbv15": "ai_ccbv15",
                    #     "xdc1/ccbv16": "ai_ccbv16",
                    #     "xdc1/ccbv17": "ai_ccbv17",
                    #     "xdc1/ccbv18": "ai_ccbv18",
                    #     "xdc1/ccbv19": "ai_ccbv19",
                    #     "xdc1/ccbv20": "ai_ccbv20",
                    #     "xdc1/ccbv21": "ai_ccbv21",
                    #     "xdc1/ccbv22": "ai_ccbv22",
                    #     "xdc1/ccbv23": "ai_ccbv23",
                    #     "xdc1/ccbv24": "ai_ccbv24",
                    #     "xdc1/ccbv25": "ai_ccbv25",
                    #     "xdc1/ccbv26": "ai_ccbv26",
                    #     "xdc1/ccbv27": "ai_ccbv27",
                    #     "xdc1/ccbv28": "ai_ccbv28",
                    #     "xdc1/ccbv29": "ai_ccbv29",
                    #     "xdc1/ccbv30": "ai_ccbv30",
                    #     "xdc1/ccbv31": "ai_ccbv31",
                    #     "xdc1/ccbv32": "ai_ccbv32",
                    #     "xdc1/ccbv33": "ai_ccbv33",
                    #     "xdc1/ccbv34": "ai_ccbv34",
                    #     "xdc1/ccbv35": "ai_ccbv35",
                    #     "xdc1/ccbv36": "ai_ccbv36",
                    #     "xdc1/ccbv37": "ai_ccbv37",
                    #     "xdc1/ccbv38": "ai_ccbv38",
                    #     "xdc1/ccbv39": "ai_ccbv39",
                    #     "xdc1/ccbv40": "ai_ccbv40",
                    #     "xdc1/co2": "ai_co2",
                    #     "xdc1/endcrown_l": "ai_endcrown_l",
                    #     "xdc1/endcrown_u": "ai_endcrown_u",
                    #     "xdc1/flcc01": "ai_flcc01",
                    #     "xdc1/flcc02": "ai_flcc02",
                    #     "xdc1/flcc03": "ai_flcc03",
                    #     "xdc1/flcc04": "ai_flcc04",
                    #     "xdc1/flcc05": "ai_flcc05",
                    #     "xdc1/flcc06": "ai_flcc06",
                    #     "xdc1/flcc07": "ai_flcc07",
                    #     "xdc1/flcc08": "ai_flcc08",
                    #     "xdc1/flcc09": "ai_flcc09",
                    #     "xdc1/flcc10": "ai_flcc10",
                    #     "xdc1/flp2l1": "ai_flp2l1",
                    #     "xdc1/flp2l2": "ai_flp2l2",
                    #     "xdc1/flp2l3": "ai_flp2l3",
                    #     "xdc1/flp2l4": "ai_flp2l4",
                    #     "xdc1/flp2u1": "ai_flp2u1",
                    #     "xdc1/flp2u2": "ai_flp2u2",
                    #     "xdc1/flp2u3": "ai_flp2u3",
                    #     "xdc1/flp2u4": "ai_flp2u4",
                    #     "xdc1/flp3l1": "ai_flp3l1",
                    #     "xdc1/flp3l4": "ai_flp3l4",
                    #     "xdc1/flp3u1": "ai_flp3u1",
                    #     "xdc1/flp3u4": "ai_flp3u4",
                    #     "xdc1/flp4l1": "ai_flp4l1",
                    #     "xdc1/flp4l4": "ai_flp4l4",
                    #     "xdc1/flp4u1": "ai_flp4u1",
                    #     "xdc1/flp4u4": "ai_flp4u4",
                    #     "xdc1/flp5l1": "ai_flp5l1",
                    #     "xdc1/flp5l4": "ai_flp5l4",
                    #     "xdc1/flp5u1": "ai_flp5u1",
                    #     "xdc1/flp5u4": "ai_flp5u4",
                    #     "xdc1/flp6l1": "ai_flp6l1",
                    #     "xdc1/flp6u1": "ai_flp6u1",
                    #     "xdc1/hene": "ai_hene",
                    #     "xdc1/incon1": "ai_incon1",
                    #     "xdc1/incon10": "ai_incon10",
                    #     "xdc1/incon2": "ai_incon2",
                    #     "xdc1/incon3": "ai_incon3",
                    #     "xdc1/incon4": "ai_incon4",
                    #     "xdc1/incon5": "ai_incon5",
                    #     "xdc1/incon6": "ai_incon6",
                    #     "xdc1/incon7": "ai_incon7",
                    #     "xdc1/incon8": "ai_incon8",
                    #     "xdc1/incon9": "ai_incon9",
                    #     "xdc1/lhorw1": "ai_lhorw1",
                    #     "xdc1/lhorw2": "ai_lhorw2",
                    #     "xdc1/lhorw3": "ai_lhorw3",
                    #     "xdc1/lhorw4": "ai_lhorw4",
                    #     "xdc1/lhorw5": "ai_lhorw5",
                    #     "xdc1/lhorw6": "ai_lhorw6",
                    #     "xdc1/loopback_in": "ai_loopback_in",
                    #     "xdc1/lvcc05": "ai_lvcc05",
                    #     "xdc1/magnetics_ok": "ai_magnetics_ok",
                    #     "xdc1/mid1": "ai_mid1",
                    #     "xdc1/mid10": "ai_mid10",
                    #     "xdc1/mid11": "ai_mid11",
                    #     "xdc1/mid12": "ai_mid12",
                    #     "xdc1/mid2": "ai_mid2",
                    #     "xdc1/mid3": "ai_mid3",
                    #     "xdc1/mid4": "ai_mid4",
                    #     "xdc1/mid5": "ai_mid5",
                    #     "xdc1/mid6": "ai_mid6",
                    #     "xdc1/mid7": "ai_mid7",
                    #     "xdc1/mid8": "ai_mid8",
                    #     "xdc1/mid9": "ai_mid9",
                    #     "xdc1/nbi_ss_i": "ai_nbi_ss_i",
                    #     "xdc1/nbi_ss_on": "ai_nbi_ss_on",
                    #     "xdc1/nbi_ss_v": "ai_nbi_ss_v",
                    #     "xdc1/nbi_sw_i": "ai_nbi_sw_i",
                    #     "xdc1/nbi_sw_on": "ai_nbi_sw_on",
                    #     "xdc1/nbi_sw_v": "ai_nbi_sw_v",
                    #     "xdc1/ntm_kick": "ai_ntm_kick",
                    #     "xdc1/obr01": "ai_obr01",
                    #     "xdc1/obr02": "ai_obr02",
                    #     "xdc1/obr03": "ai_obr03",
                    #     "xdc1/obr04": "ai_obr04",
                    #     "xdc1/obr05": "ai_obr05",
                    #     "xdc1/obr06": "ai_obr06",
                    #     "xdc1/obr07": "ai_obr07",
                    #     "xdc1/obr08": "ai_obr08",
                    #     "xdc1/obr09": "ai_obr09",
                    #     "xdc1/obr10": "ai_obr10",
                    #     "xdc1/obr11": "ai_obr11",
                    #     "xdc1/obr12": "ai_obr12",
                    #     "xdc1/obr13": "ai_obr13",
                    #     "xdc1/obr14": "ai_obr14",
                    #     "xdc1/obr15": "ai_obr15",
                    #     "xdc1/obr16": "ai_obr16",
                    #     "xdc1/obr17": "ai_obr17",
                    #     "xdc1/obr18": "ai_obr18",
                    #     "xdc1/obr19": "ai_obr19",
                    #     "xdc1/obv01": "ai_obv01",
                    #     "xdc1/obv02": "ai_obv02",
                    #     "xdc1/obv03": "ai_obv03",
                    #     "xdc1/obv04": "ai_obv04",
                    #     "xdc1/obv05": "ai_obv05",
                    #     "xdc1/obv06": "ai_obv06",
                    #     "xdc1/obv07": "ai_obv07",
                    #     "xdc1/obv08": "ai_obv08",
                    #     "xdc1/obv09": "ai_obv09",
                    #     "xdc1/obv10": "ai_obv10",
                    #     "xdc1/obv11": "ai_obv11",
                    #     "xdc1/obv12": "ai_obv12",
                    #     "xdc1/obv13": "ai_obv13",
                    #     "xdc1/obv14": "ai_obv14",
                    #     "xdc1/obv15": "ai_obv15",
                    #     "xdc1/obv16": "ai_obv16",
                    #     "xdc1/obv17": "ai_obv17",
                    #     "xdc1/obv18": "ai_obv18",
                    #     "xdc1/obv19": "ai_obv19",
                    #     "xdc1/p2larm1": "ai_p2larm1",
                    #     "xdc1/p2larm2": "ai_p2larm2",
                    #     "xdc1/p2larm3": "ai_p2larm3",
                    #     "xdc1/p2l_case": "ai_p2l_case",
                    #     "xdc1/p2ldivpl1": "ai_p2ldivpl1",
                    #     "xdc1/p2ldivpl2": "ai_p2ldivpl2",
                    #     "xdc1/p2uarm1": "ai_p2uarm1",
                    #     "xdc1/p2uarm2": "ai_p2uarm2",
                    #     "xdc1/p2uarm3": "ai_p2uarm3",
                    #     "xdc1/p2u_case": "ai_p2u_case",
                    #     "xdc1/p2udivpl1": "ai_p2udivpl1",
                    #     "xdc1/p2udivpl2": "ai_p2udivpl2",
                    #     "xdc1/p3l_case": "ai_p3l_case",
                    #     "xdc1/p3u_case": "ai_p3u_case",
                    #     "xdc1/p4l_case": "ai_p4l_case",
                    #     "xdc1/p4u_case": "ai_p4u_case",
                    #     "xdc1/p5l_case": "ai_p5l_case",
                    #     "xdc1/p5u_case": "ai_p5u_case",
                    #     "xdc1/p6l_case": "ai_p6l_case",
                    #     "xdc1/p6u_case": "ai_p6u_case",
                    #     "xdc1/plasma_current": "ai_plasma_current",
                    #     "xdc1/power_on": "ai_power_on",
                    #     "xdc1/ring1": "ai_ring1",
                    #     "xdc1/ring10": "ai_ring10",
                    #     "xdc1/ring2": "ai_ring2",
                    #     "xdc1/ring3": "ai_ring3",
                    #     "xdc1/ring4": "ai_ring4",
                    #     "xdc1/ring5": "ai_ring5",
                    #     "xdc1/ring6": "ai_ring6",
                    #     "xdc1/ring7": "ai_ring7",
                    #     "xdc1/ring8": "ai_ring8",
                    #     "xdc1/ring9": "ai_ring9",
                    #     "xdc1/rodgr1": "ai_rodgr1",
                    #     "xdc1/rodgr10": "ai_rodgr10",
                    #     "xdc1/rodgr11": "ai_rodgr11",
                    #     "xdc1/rodgr12": "ai_rodgr12",
                    #     "xdc1/rodgr2": "ai_rodgr2",
                    #     "xdc1/rodgr3": "ai_rodgr3",
                    #     "xdc1/rodgr4": "ai_rodgr4",
                    #     "xdc1/rodgr5": "ai_rodgr5",
                    #     "xdc1/rodgr6": "ai_rodgr6",
                    #     "xdc1/rodgr7": "ai_rodgr7",
                    #     "xdc1/rodgr8": "ai_rodgr8",
                    #     "xdc1/rodgr9": "ai_rodgr9",
                    #     "xdc1/rogext_efcc02": "ai_rogext_efcc02",
                    #     "xdc1/rogext_efcc05": "ai_rogext_efcc05",
                    #     "xdc1/rogext_efps": "ai_rogext_efps",
                    #     "xdc1/rogext_p2li": "ai_rogext_p2li",
                    #     "xdc1/rogext_p2lo": "ai_rogext_p2lo",
                    #     "xdc1/rogext_p2ui": "ai_rogext_p2ui",
                    #     "xdc1/rogext_p2uo": "ai_rogext_p2uo",
                    #     "xdc1/rogext_p3l": "ai_rogext_p3l",
                    #     "xdc1/rogext_p3u": "ai_rogext_p3u",
                    #     "xdc1/rogext_p4l": "ai_rogext_p4l",
                    #     "xdc1/rogext_p4u": "ai_rogext_p4u",
                    #     "xdc1/rogext_p5l": "ai_rogext_p5l",
                    #     "xdc1/rogext_p5u": "ai_rogext_p5u",
                    #     "xdc1/rogext_p6a": "ai_rogext_p6a",
                    #     "xdc1/rogext_p6b": "ai_rogext_p6b",
                    #     "xdc1/rogext_sol": "ai_rogext_sol",
                    #     "xdc1/rogext_test": "ai_rogext_test",
                    #     "xdc1/rog_ip02_1": "ai_rog_ip02_1",
                    #     "xdc1/rog_ip02_2": "ai_rog_ip02_2",
                    #     "xdc1/rog_ip02_3": "ai_rog_ip02_3",
                    #     "xdc1/rog_ip02_4": "ai_rog_ip02_4",
                    #     "xdc1/rog_p2l_1": "ai_rog_p2l_1",
                    #     "xdc1/rog_p2u_1": "ai_rog_p2u_1",
                    #     "xdc1/rog_p3l_2": "ai_rog_p3l_2",
                    #     "xdc1/rog_p3u_2": "ai_rog_p3u_2",
                    #     "xdc1/rog_p4l_2": "ai_rog_p4l_2",
                    #     "xdc1/rog_p4u_2": "ai_rog_p4u_2",
                    #     "xdc1/rog_p5l_2": "ai_rog_p5l_2",
                    #     "xdc1/rog_p5u_2": "ai_rog_p5u_2",
                    #     "xdc1/rog_p6l_2": "ai_rog_p6l_2",
                    #     "xdc1/rog_p6u_2": "ai_rog_p6u_2",
                    #     "xdc1/r_outer": "ai_r_outer",
                    #     "xdc1/testch1": "ai_testch1",
                    #     "xdc1/testch2": "ai_testch2",
                    #     "xdc1/testch3": "ai_testch3",
                    #     "xdc1/tf_current": "ai_tf_current",
                    #     "xdc1/topcol": "ai_topcol",
                    #     "xdc1/uhorw1": "ai_uhorw1",
                    #     "xdc1/uhorw2": "ai_uhorw2",
                    #     "xdc1/uhorw3": "ai_uhorw3",
                    #     "xdc1/uhorw4": "ai_uhorw4",
                    #     "xdc1/uhorw5": "ai_uhorw5",
                    #     "xdc1/uhorw6": "ai_uhorw6",
                    #     "xdc1/vertw1": "ai_vertw1",
                    #     "xdc1/vertw2": "ai_vertw2",
                    #     "xdc1/vertw3": "ai_vertw3",
                    #     "xdc1/vertw4": "ai_vertw4",
                    #     "xdc1/vertw5": "ai_vertw5",
                    #     "xdc1/vertw6": "ai_vertw6",
                    #     "xdc1/vertw7": "ai_vertw7",
                    #     "xdc1/vertw8": "ai_vertw8",
                    #     "xdc1/watchdog_ok": "ai_watchdog_ok",
                    #     "xdc/bc11_drive": "ao_bc11_drive",
                    #     "xdc/bc5_drive": "ao_bc5_drive",
                    #     "xdc/camera_ok": "ai_raw_camera_ok",
                    #     "xdc/ccbv01": "ai_raw_ccbv01",
                    #     "xdc/ccbv02": "ai_raw_ccbv02",
                    #     "xdc/ccbv03": "ai_raw_ccbv03",
                    #     "xdc/ccbv04": "ai_raw_ccbv04",
                    #     "xdc/ccbv05": "ai_raw_ccbv05",
                    #     "xdc/ccbv06": "ai_raw_ccbv06",
                    #     "xdc/ccbv07": "ai_raw_ccbv07",
                    #     "xdc/ccbv08": "ai_raw_ccbv08",
                    #     "xdc/ccbv09": "ai_raw_ccbv09",
                    #     "xdc/ccbv10": "ai_raw_ccbv10",
                    #     "xdc/ccbv11": "ai_raw_ccbv11",
                    #     "xdc/ccbv12": "ai_raw_ccbv12",
                    #     "xdc/ccbv13": "ai_raw_ccbv13",
                    #     "xdc/ccbv14": "ai_raw_ccbv14",
                    #     "xdc/ccbv15": "ai_raw_ccbv15",
                    #     "xdc/ccbv16": "ai_raw_ccbv16",
                    #     "xdc/ccbv17": "ai_raw_ccbv17",
                    #     "xdc/ccbv18": "ai_raw_ccbv18",
                    #     "xdc/ccbv19": "ai_raw_ccbv19",
                    #     "xdc/ccbv20": "ai_raw_ccbv20",
                    #     "xdc/ccbv21": "ai_raw_ccbv21",
                    #     "xdc/ccbv22": "ai_raw_ccbv22",
                    #     "xdc/ccbv23": "ai_raw_ccbv23",
                    #     "xdc/ccbv24": "ai_raw_ccbv24",
                    #     "xdc/ccbv25": "ai_raw_ccbv25",
                    #     "xdc/ccbv26": "ai_raw_ccbv26",
                    #     "xdc/ccbv27": "ai_raw_ccbv27",
                    #     "xdc/ccbv28": "ai_raw_ccbv28",
                    #     "xdc/ccbv29": "ai_raw_ccbv29",
                    #     "xdc/ccbv30": "ai_raw_ccbv30",
                    #     "xdc/ccbv31": "ai_raw_ccbv31",
                    #     "xdc/ccbv32": "ai_raw_ccbv32",
                    #     "xdc/ccbv33": "ai_raw_ccbv33",
                    #     "xdc/ccbv34": "ai_raw_ccbv34",
                    #     "xdc/ccbv35": "ai_raw_ccbv35",
                    #     "xdc/ccbv36": "ai_raw_ccbv36",
                    #     "xdc/ccbv37": "ai_raw_ccbv37",
                    #     "xdc/ccbv38": "ai_raw_ccbv38",
                    #     "xdc/ccbv39": "ai_raw_ccbv39",
                    #     "xdc/ccbv40": "ai_raw_ccbv40",
                    #     "xdc/co2": "ai_raw_co2",
                    #     "xdc/density_e_nel": "density_e_nel",
                    #     "xdc/density_p_nel": "density_p_nel",
                    #     "xdc/density_s_densok": "density_s_densok",
                    #     "xdc/density_s_nel": "density_s_nel",
                    #     "xdc/density_s_test1": "density_s_test1",
                    #     "xdc/density_s_test2": "density_s_test2",
                    #     "xdc/density_s_test3": "density_s_test3",
                    #     "xdc/density_t_nelref": "density_t_nelref",
                    #     "xdc/dmv_trigger": "do_dmv_trigger",
                    #     "xdc/do_13": "do_do_13",
                    #     "xdc/do_14": "do_do_14",
                    #     "xdc/do_15": "do_do_15",
                    #     "xdc/do_16": "do_do_16",
                    #     "xdc/eceleste": "ao_eceleste",
                    #     "xdc/efc_0208_drive": "ao_efc_0208_drive",
                    #     "xdc/efc_0511_drive": "ao_efc_0511_drive",
                    #     "xdc/ef_e_err0208": "ef_e_err0208",
                    #     "xdc/ef_e_err0511": "ef_e_err0511",
                    #     "xdc/ef_f_0208": "ef_f_0208",
                    #     "xdc/ef_f_0511": "ef_f_0511",
                    #     "xdc/ef_s_test1": "ef_s_test1",
                    #     "xdc/ef_s_test2": "ef_s_test2",
                    #     "xdc/ef_t_iref0208": "ef_t_iref0208",
                    #     "xdc/ef_t_iref0208_ff": "ef_t_iref0208_ff",
                    #     "xdc/ef_t_iref0511": "ef_t_iref0511",
                    #     "xdc/ef_t_iref0511_ff": "ef_t_iref0511_ff",
                    #     "xdc/ef_t_vref0208": "ef_t_vref0208",
                    #     "xdc/ef_t_vref0208_ff": "ef_t_vref0208_ff",
                    #     "xdc/ef_t_vref0511": "ef_t_vref0511",
                    #     "xdc/ef_t_vref0511_ff": "ef_t_vref0511_ff",
                    #     "xdc/elm_a_drive": "elm_a_drive",
                    #     "xdc/elm_b_drive": "elm_b_drive",
                    #     "xdc/elm_c_drive": "elm_c_drive",
                    #     "xdc/elm_d_drive": "elm_d_drive",
                    #     "xdc/elm_f_a": "elm_f_a",
                    #     "xdc/elm_f_b": "elm_f_b",
                    #     "xdc/elm_f_c": "elm_f_c",
                    #     "xdc/elm_f_d": "elm_f_d",
                    #     "xdc/elm_t_irefa": "elm_t_irefa",
                    #     "xdc/elm_t_irefb": "elm_t_irefb",
                    #     "xdc/elm_t_irefc": "elm_t_irefc",
                    #     "xdc/elm_t_irefd": "elm_t_irefd",
                    #     "xdc/equil_s_13dlt": "equil_s_13dlt",
                    #     "xdc/equil_s_13err": "equil_s_13err",
                    #     "xdc/equil_s_13jtch": "equil_s_13jtch",
                    #     "xdc/equil_s_13ltch": "equil_s_13ltch",
                    #     "xdc/equil_s_13rtch": "equil_s_13rtch",
                    #     "xdc/equil_s_13ztch": "equil_s_13ztch",
                    #     "xdc/equil_s_14dlt": "equil_s_14dlt",
                    #     "xdc/equil_s_14err": "equil_s_14err",
                    #     "xdc/equil_s_14jtch": "equil_s_14jtch",
                    #     "xdc/equil_s_14ltch": "equil_s_14ltch",
                    #     "xdc/equil_s_14rtch": "equil_s_14rtch",
                    #     "xdc/equil_s_14ztch": "equil_s_14ztch",
                    #     "xdc/equil_s_15dlt": "equil_s_15dlt",
                    #     "xdc/equil_s_15err": "equil_s_15err",
                    #     "xdc/equil_s_15jtch": "equil_s_15jtch",
                    #     "xdc/equil_s_15ltch": "equil_s_15ltch",
                    #     "xdc/equil_s_15rtch": "equil_s_15rtch",
                    #     "xdc/equil_s_15ztch": "equil_s_15ztch",
                    #     "xdc/equil_s_betan": "equil_s_betan",
                    #     "xdc/equil_s_betap": "equil_s_betap",
                    #     "xdc/equil_s_betat": "equil_s_betat",
                    #     "xdc/equil_s_deltaz": "equil_s_deltaz",
                    #     "xdc/equil_s_flerror": "equil_s_flerror",
                    #     "xdc/equil_s_flitimb": "equil_s_flitimb",
                    #     "xdc/equil_s_flitime": "equil_s_flitime",
                    #     "xdc/equil_s_g1br": "equil_s_g1br",
                    #     "xdc/equil_s_g1bz": "equil_s_g1bz",
                    #     "xdc/equil_s_g1cthet": "equil_s_g1cthet",
                    #     "xdc/equil_s_g1dr": "equil_s_g1dr",
                    #     "xdc/equil_s_g1dz": "equil_s_g1dz",
                    #     "xdc/equil_s_g1err": "equil_s_g1err",
                    #     "xdc/equil_s_g1i": "equil_s_g1i",
                    #     "xdc/equil_s_g1j": "equil_s_g1j",
                    #     "xdc/equil_s_g1psi": "equil_s_g1psi",
                    #     "xdc/equil_s_g1rx": "equil_s_g1rx",
                    #     "xdc/equil_s_g1rxt": "equil_s_g1rxt",
                    #     "xdc/equil_s_g1zx": "equil_s_g1zx",
                    #     "xdc/equil_s_g1zxt": "equil_s_g1zxt",
                    #     "xdc/equil_s_g2br": "equil_s_g2br",
                    #     "xdc/equil_s_g2bz": "equil_s_g2bz",
                    #     "xdc/equil_s_g2cthet": "equil_s_g2cthet",
                    #     "xdc/equil_s_g2dr": "equil_s_g2dr",
                    #     "xdc/equil_s_g2dz": "equil_s_g2dz",
                    #     "xdc/equil_s_g2err": "equil_s_g2err",
                    #     "xdc/equil_s_g2i": "equil_s_g2i",
                    #     "xdc/equil_s_g2j": "equil_s_g2j",
                    #     "xdc/equil_s_g2psi": "equil_s_g2psi",
                    #     "xdc/equil_s_g2rx": "equil_s_g2rx",
                    #     "xdc/equil_s_g2rxt": "equil_s_g2rxt",
                    #     "xdc/equil_s_g2zx": "equil_s_g2zx",
                    #     "xdc/equil_s_g2zxt": "equil_s_g2zxt",
                    #     "xdc/equil_s_ipmeas": "equil_s_ipmeas",
                    #     "xdc/equil_s_ipmhd": "equil_s_ipmhd",
                    #     "xdc/equil_s_isonoff": "equil_s_isonoff",
                    #     "xdc/equil_s_li": "equil_s_li",
                    #     "xdc/equil_s_li3": "equil_s_li3",
                    #     "xdc/equil_s_lierr": "equil_s_lierr",
                    #     "xdc/equil_s_psibdy1": "equil_s_psibdy1",
                    #     "xdc/equil_s_psibdy2": "equil_s_psibdy2",
                    #     "xdc/equil_s_psimag": "equil_s_psimag",
                    #     "xdc/equil_s_psiref": "equil_s_psiref",
                    #     "xdc/equil_s_psnqmnb": "equil_s_psnqmnb",
                    #     "xdc/equil_s_q0b": "equil_s_q0b",
                    #     "xdc/equil_s_q97b": "equil_s_q97b",
                    #     "xdc/equil_s_qactb": "equil_s_qactb",
                    #     "xdc/equil_s_qminb": "equil_s_qminb",
                    #     "xdc/equil_s_ref1": "equil_s_ref1",
                    #     "xdc/equil_s_ref2": "equil_s_ref2",
                    #     "xdc/equil_s_ref3": "equil_s_ref3",
                    #     "xdc/equil_s_ref4": "equil_s_ref4",
                    #     "xdc/equil_s_rqactb": "equil_s_rqactb",
                    #     "xdc/equil_s_rsur97b": "equil_s_rsur97b",
                    #     "xdc/equil_s_seg01": "equil_s_seg01",
                    #     "xdc/equil_s_seg01at": "equil_s_seg01at",
                    #     "xdc/equil_s_seg01rt": "equil_s_seg01rt",
                    #     "xdc/equil_s_seg01zt": "equil_s_seg01zt",
                    #     "xdc/equil_s_seg02": "equil_s_seg02",
                    #     "xdc/equil_s_seg02at": "equil_s_seg02at",
                    #     "xdc/equil_s_seg02rt": "equil_s_seg02rt",
                    #     "xdc/equil_s_seg02zt": "equil_s_seg02zt",
                    #     "xdc/equil_s_seg03": "equil_s_seg03",
                    #     "xdc/equil_s_seg03at": "equil_s_seg03at",
                    #     "xdc/equil_s_seg03rt": "equil_s_seg03rt",
                    #     "xdc/equil_s_seg03zt": "equil_s_seg03zt",
                    #     "xdc/equil_s_seg04": "equil_s_seg04",
                    #     "xdc/equil_s_seg04at": "equil_s_seg04at",
                    #     "xdc/equil_s_seg04rt": "equil_s_seg04rt",
                    #     "xdc/equil_s_seg04zt": "equil_s_seg04zt",
                    #     "xdc/equil_s_seg05": "equil_s_seg05",
                    #     "xdc/equil_s_seg05at": "equil_s_seg05at",
                    #     "xdc/equil_s_seg05rt": "equil_s_seg05rt",
                    #     "xdc/equil_s_seg05zt": "equil_s_seg05zt",
                    #     "xdc/equil_s_seg06": "equil_s_seg06",
                    #     "xdc/equil_s_seg06at": "equil_s_seg06at",
                    #     "xdc/equil_s_seg06rt": "equil_s_seg06rt",
                    #     "xdc/equil_s_seg06zt": "equil_s_seg06zt",
                    #     "xdc/equil_s_seg07": "equil_s_seg07",
                    #     "xdc/equil_s_seg07at": "equil_s_seg07at",
                    #     "xdc/equil_s_seg07rt": "equil_s_seg07rt",
                    #     "xdc/equil_s_seg07zt": "equil_s_seg07zt",
                    #     "xdc/equil_s_seg08": "equil_s_seg08",
                    #     "xdc/equil_s_seg08at": "equil_s_seg08at",
                    #     "xdc/equil_s_seg08rt": "equil_s_seg08rt",
                    #     "xdc/equil_s_seg08zt": "equil_s_seg08zt",
                    #     "xdc/equil_s_seg09": "equil_s_seg09",
                    #     "xdc/equil_s_seg09at": "equil_s_seg09at",
                    #     "xdc/equil_s_seg09rt": "equil_s_seg09rt",
                    #     "xdc/equil_s_seg09zt": "equil_s_seg09zt",
                    #     "xdc/equil_s_seg10": "equil_s_seg10",
                    #     "xdc/equil_s_seg10at": "equil_s_seg10at",
                    #     "xdc/equil_s_seg10rt": "equil_s_seg10rt",
                    #     "xdc/equil_s_seg10zt": "equil_s_seg10zt",
                    #     "xdc/equil_s_seg11": "equil_s_seg11",
                    #     "xdc/equil_s_seg11at": "equil_s_seg11at",
                    #     "xdc/equil_s_seg11rt": "equil_s_seg11rt",
                    #     "xdc/equil_s_seg11zt": "equil_s_seg11zt",
                    #     "xdc/equil_s_seg12": "equil_s_seg12",
                    #     "xdc/equil_s_seg12at": "equil_s_seg12at",
                    #     "xdc/equil_s_seg12rt": "equil_s_seg12rt",
                    #     "xdc/equil_s_seg12zt": "equil_s_seg12zt",
                    #     "xdc/equil_s_seg13": "equil_s_seg13",
                    #     "xdc/equil_s_seg13at": "equil_s_seg13at",
                    #     "xdc/equil_s_seg13rt": "equil_s_seg13rt",
                    #     "xdc/equil_s_seg13zt": "equil_s_seg13zt",
                    #     "xdc/equil_s_seg14": "equil_s_seg14",
                    #     "xdc/equil_s_seg14at": "equil_s_seg14at",
                    #     "xdc/equil_s_seg14rt": "equil_s_seg14rt",
                    #     "xdc/equil_s_seg14zt": "equil_s_seg14zt",
                    #     "xdc/equil_s_seg15": "equil_s_seg15",
                    #     "xdc/equil_s_seg15at": "equil_s_seg15at",
                    #     "xdc/equil_s_seg15rt": "equil_s_seg15rt",
                    #     "xdc/equil_s_seg15zt": "equil_s_seg15zt",
                    #     "xdc/equil_s_seg16": "equil_s_seg16",
                    #     "xdc/equil_s_seg16at": "equil_s_seg16at",
                    #     "xdc/equil_s_seg16rt": "equil_s_seg16rt",
                    #     "xdc/equil_s_seg16zt": "equil_s_seg16zt",
                    #     "xdc/equil_s_seg17": "equil_s_seg17",
                    #     "xdc/equil_s_seg17at": "equil_s_seg17at",
                    #     "xdc/equil_s_seg17rt": "equil_s_seg17rt",
                    #     "xdc/equil_s_seg17zt": "equil_s_seg17zt",
                    #     "xdc/equil_s_seg18": "equil_s_seg18",
                    #     "xdc/equil_s_seg18at": "equil_s_seg18at",
                    #     "xdc/equil_s_seg18rt": "equil_s_seg18rt",
                    #     "xdc/equil_s_seg18zt": "equil_s_seg18zt",
                    #     "xdc/equil_s_segb01": "equil_s_segb01",
                    #     "xdc/equil_s_segb02": "equil_s_segb02",
                    #     "xdc/equil_s_segb03": "equil_s_segb03",
                    #     "xdc/equil_s_segb04": "equil_s_segb04",
                    #     "xdc/equil_s_segb05": "equil_s_segb05",
                    #     "xdc/equil_s_segb06": "equil_s_segb06",
                    #     "xdc/equil_s_segb07": "equil_s_segb07",
                    #     "xdc/equil_s_segb08": "equil_s_segb08",
                    #     "xdc/equil_s_segb09": "equil_s_segb09",
                    #     "xdc/equil_s_segb10": "equil_s_segb10",
                    #     "xdc/equil_s_segb11": "equil_s_segb11",
                    #     "xdc/equil_s_segb12": "equil_s_segb12",
                    #     "xdc/equil_s_segb13": "equil_s_segb13",
                    #     "xdc/equil_s_segb14": "equil_s_segb14",
                    #     "xdc/equil_s_segb15": "equil_s_segb15",
                    #     "xdc/equil_s_segb16": "equil_s_segb16",
                    #     "xdc/equil_s_segb17": "equil_s_segb17",
                    #     "xdc/equil_s_segb18": "equil_s_segb18",
                    #     "xdc/equil_s_siberr": "equil_s_siberr",
                    #     "xdc/equil_s_sipmeas": "equil_s_sipmeas",
                    #     "xdc/equil_s_slerror": "equil_s_slerror",
                    #     "xdc/equil_s_slitime": "equil_s_slitime",
                    #     "xdc/equil_s_slpsib": "equil_s_slpsib",
                    #     "xdc/equil_s_slrmax": "equil_s_slrmax",
                    #     "xdc/equil_s_slrmin": "equil_s_slrmin",
                    #     "xdc/equil_s_slrun": "equil_s_slrun",
                    #     "xdc/equil_s_slzmax": "equil_s_slzmax",
                    #     "xdc/equil_s_slzmin": "equil_s_slzmin",
                    #     "xdc/equil_s_werr": "equil_s_werr",
                    #     "xdc/equil_s_wmhd": "equil_s_wmhd",
                    #     "xdc/equil_s_zcur": "equil_s_zcur",
                    #     "xdc/equil_s_zcurval": "equil_s_zcurval",
                    #     "xdc/equil_t_ctlim": "equil_t_ctlim",
                    #     "xdc/equil_t_dlmlim": "equil_t_dlmlim",
                    #     "xdc/equil_t_flinpub": "equil_t_flinpub",
                    #     "xdc/equil_t_flinput": "equil_t_flinput",
                    #     "xdc/equil_t_g1mdiff": "equil_t_g1mdiff",
                    #     "xdc/equil_t_g1mstep": "equil_t_g1mstep",
                    #     "xdc/equil_t_g2mdiff": "equil_t_g2mdiff",
                    #     "xdc/equil_t_g2mstep": "equil_t_g2mstep",
                    #     "xdc/equil_t_grid1u": "equil_t_grid1u",
                    #     "xdc/equil_t_grid2u": "equil_t_grid2u",
                    #     "xdc/equil_t_refpt1": "equil_t_refpt1",
                    #     "xdc/equil_t_refpt2": "equil_t_refpt2",
                    #     "xdc/equil_t_refpt3": "equil_t_refpt3",
                    #     "xdc/equil_t_refpt4": "equil_t_refpt4",
                    #     "xdc/equil_t_rgrid1": "equil_t_rgrid1",
                    #     "xdc/equil_t_rgrid2": "equil_t_rgrid2",
                    #     "xdc/equil_t_rmax": "equil_t_rmax",
                    #     "xdc/equil_t_rmin": "equil_t_rmin",
                    #     "xdc/equil_t_saminta": "equil_t_saminta",
                    #     "xdc/equil_t_samintb": "equil_t_samintb",
                    #     "xdc/equil_t_seg01": "equil_t_seg01",
                    #     "xdc/equil_t_seg01u": "equil_t_seg01u",
                    #     "xdc/equil_t_seg02": "equil_t_seg02",
                    #     "xdc/equil_t_seg02u": "equil_t_seg02u",
                    #     "xdc/equil_t_seg03": "equil_t_seg03",
                    #     "xdc/equil_t_seg03u": "equil_t_seg03u",
                    #     "xdc/equil_t_seg04": "equil_t_seg04",
                    #     "xdc/equil_t_seg04u": "equil_t_seg04u",
                    #     "xdc/equil_t_seg05": "equil_t_seg05",
                    #     "xdc/equil_t_seg05u": "equil_t_seg05u",
                    #     "xdc/equil_t_seg06": "equil_t_seg06",
                    #     "xdc/equil_t_seg06u": "equil_t_seg06u",
                    #     "xdc/equil_t_seg07": "equil_t_seg07",
                    #     "xdc/equil_t_seg07u": "equil_t_seg07u",
                    #     "xdc/equil_t_seg08": "equil_t_seg08",
                    #     "xdc/equil_t_seg08u": "equil_t_seg08u",
                    #     "xdc/equil_t_seg09": "equil_t_seg09",
                    #     "xdc/equil_t_seg09u": "equil_t_seg09u",
                    #     "xdc/equil_t_seg10": "equil_t_seg10",
                    #     "xdc/equil_t_seg10u": "equil_t_seg10u",
                    #     "xdc/equil_t_seg11": "equil_t_seg11",
                    #     "xdc/equil_t_seg11u": "equil_t_seg11u",
                    #     "xdc/equil_t_seg12": "equil_t_seg12",
                    #     "xdc/equil_t_seg12u": "equil_t_seg12u",
                    #     "xdc/equil_t_seg13": "equil_t_seg13",
                    #     "xdc/equil_t_seg13u": "equil_t_seg13u",
                    #     "xdc/equil_t_seg14": "equil_t_seg14",
                    #     "xdc/equil_t_seg14u": "equil_t_seg14u",
                    #     "xdc/equil_t_seg15": "equil_t_seg15",
                    #     "xdc/equil_t_seg15u": "equil_t_seg15u",
                    #     "xdc/equil_t_seg16": "equil_t_seg16",
                    #     "xdc/equil_t_seg16u": "equil_t_seg16u",
                    #     "xdc/equil_t_seg17": "equil_t_seg17",
                    #     "xdc/equil_t_seg17u": "equil_t_seg17u",
                    #     "xdc/equil_t_seg18": "equil_t_seg18",
                    #     "xdc/equil_t_seg18u": "equil_t_seg18u",
                    #     "xdc/equil_t_snapa": "equil_t_snapa",
                    #     "xdc/equil_t_snapb": "equil_t_snapb",
                    #     "xdc/equil_t_usex": "equil_t_usex",
                    #     "xdc/equil_t_xrmaxb": "equil_t_xrmaxb",
                    #     "xdc/equil_t_xrmaxt": "equil_t_xrmaxt",
                    #     "xdc/equil_t_xrminb": "equil_t_xrminb",
                    #     "xdc/equil_t_xrmint": "equil_t_xrmint",
                    #     "xdc/equil_t_xzmaxb": "equil_t_xzmaxb",
                    #     "xdc/equil_t_xzmaxt": "equil_t_xzmaxt",
                    #     "xdc/equil_t_xzminb": "equil_t_xzminb",
                    #     "xdc/equil_t_xzmint": "equil_t_xzmint",
                    #     "xdc/equil_t_zgrid1": "equil_t_zgrid1",
                    #     "xdc/equil_t_zgrid2": "equil_t_zgrid2",
                    #     "xdc/equil_t_zmax": "equil_t_zmax",
                    #     "xdc/equil_t_zmin": "equil_t_zmin",
                    #     "xdc/flcc01": "ai_raw_flcc01",
                    #     "xdc/flcc02": "ai_raw_flcc02",
                    #     "xdc/flcc03": "ai_raw_flcc03",
                    #     "xdc/flcc04": "ai_raw_flcc04",
                    #     "xdc/flcc05": "ai_raw_flcc05",
                    #     "xdc/flcc06": "ai_raw_flcc06",
                    #     "xdc/flcc07": "ai_raw_flcc07",
                    #     "xdc/flcc08": "ai_raw_flcc08",
                    #     "xdc/flcc09": "ai_raw_flcc09",
                    #     "xdc/flcc10": "ai_raw_flcc10",
                    #     "xdc/flp2l1": "ai_raw_flp2l1",
                    #     "xdc/flp2l2": "ai_raw_flp2l2",
                    #     "xdc/flp2l3": "ai_raw_flp2l3",
                    #     "xdc/flp2l4": "ai_raw_flp2l4",
                    #     "xdc/flp2u1": "ai_raw_flp2u1",
                    #     "xdc/flp2u2": "ai_raw_flp2u2",
                    #     "xdc/flp2u3": "ai_raw_flp2u3",
                    #     "xdc/flp2u4": "ai_raw_flp2u4",
                    #     "xdc/flp3l1": "ai_raw_flp3l1",
                    #     "xdc/flp3l4": "ai_raw_flp3l4",
                    #     "xdc/flp3u1": "ai_raw_flp3u1",
                    #     "xdc/flp3u4": "ai_raw_flp3u4",
                    #     "xdc/flp4l1": "ai_raw_flp4l1",
                    #     "xdc/flp4l4": "ai_raw_flp4l4",
                    #     "xdc/flp4u1": "ai_raw_flp4u1",
                    #     "xdc/flp4u4": "ai_raw_flp4u4",
                    #     "xdc/flp5l1": "ai_raw_flp5l1",
                    #     "xdc/flp5l4": "ai_raw_flp5l4",
                    #     "xdc/flp5u1": "ai_raw_flp5u1",
                    #     "xdc/flp5u4": "ai_raw_flp5u4",
                    #     "xdc/flp6l1": "ai_raw_flp6l1",
                    #     "xdc/flp6u1": "ai_raw_flp6u1",
                    #     "xdc/gas_f_bc11": "gas_f_bc11",
                    #     "xdc/gas_f_bc5": "gas_f_bc5",
                    #     "xdc/gas_f_ecel": "gas_f_ecel",
                    #     "xdc/gas_f_hecc": "gas_f_hecc",
                    #     "xdc/gas_f_hfs": "gas_f_hfs",
                    #     "xdc/gas_f_hl1": "gas_f_hl1",
                    #     "xdc/gas_f_hl11": "gas_f_hl11",
                    #     "xdc/gas_f_hm12a": "gas_f_hm12a",
                    #     "xdc/gas_f_hm12b": "gas_f_hm12b",
                    #     "xdc/gas_f_hu11": "gas_f_hu11",
                    #     "xdc/gas_f_hu6": "gas_f_hu6",
                    #     "xdc/gas_f_hu8": "gas_f_hu8",
                    #     "xdc/gas_f_ibfla": "gas_f_ibfla",
                    #     "xdc/gas_f_ibflb": "gas_f_ibflb",
                    #     "xdc/gas_f_ibfua": "gas_f_ibfua",
                    #     "xdc/gas_f_ibfub": "gas_f_ibfub",
                    #     "xdc/gas_f_ibil": "gas_f_ibil",
                    #     "xdc/gas_f_tc11": "gas_f_tc11",
                    #     "xdc/gas_f_tc5a": "gas_f_tc5a",
                    #     "xdc/gas_f_tc5b": "gas_f_tc5b",
                    #     "xdc/gas_s_fbmode": "gas_s_fbmode",
                    #     "xdc/gas_s_last_fbmode": "gas_s_last_fbmode",
                    #     "xdc/gas_s_targflow": "gas_s_targflow",
                    #     "xdc/gas_t_ecelmod": "gas_t_ecelmod",
                    #     "xdc/gas_t_fbmode": "gas_t_fbmode",
                    #     "xdc/gas_t_flref": "gas_t_flref",
                    #     "xdc/gas_t_g1": "gas_t_g1",
                    #     "xdc/gas_t_g10": "gas_t_g10",
                    #     "xdc/gas_t_g11": "gas_t_g11",
                    #     "xdc/gas_t_g11sel": "gas_t_g11sel",
                    #     "xdc/gas_t_g12": "gas_t_g12",
                    #     "xdc/gas_t_g12sel": "gas_t_g12sel",
                    #     "xdc/gas_t_g1sel": "gas_t_g1sel",
                    #     "xdc/gas_t_g2": "gas_t_g2",
                    #     "xdc/gas_t_g2sel": "gas_t_g2sel",
                    #     "xdc/gas_t_g3": "gas_t_g3",
                    #     "xdc/gas_t_g3sel": "gas_t_g3sel",
                    #     "xdc/gas_t_g4": "gas_t_g4",
                    #     "xdc/gas_t_g4sel": "gas_t_g4sel",
                    #     "xdc/gas_t_g5": "gas_t_g5",
                    #     "xdc/gas_t_g6": "gas_t_g6",
                    #     "xdc/gas_t_g7": "gas_t_g7",
                    #     "xdc/gas_t_g9": "gas_t_g9",
                    #     "xdc/gas_t_gcal": "gas_t_gcal",
                    #     "xdc/gas_t_goffs": "gas_t_goffs",
                    #     "xdc/gas_t_maxflow": "gas_t_maxflow",
                    #     "xdc/hecc_drive": "ao_hecc_drive",
                    #     "xdc/hene": "ai_raw_hene",
                    #     "xdc/hfs_puff": "ao_hfs_puff",
                    #     "xdc/hl11_drive": "ao_hl11_drive",
                    #     "xdc/hl1_drive": "ao_hl1_drive",
                    #     "xdc/hm12a_drive": "ao_hm12a_drive",
                    #     "xdc/hm12b_drive": "ao_hm12b_drive",
                    #     "xdc/hu11_drive": "ao_hu11_drive",
                    #     "xdc/hu6_drive": "ao_hu6_drive",
                    #     "xdc/hu8_drive": "ao_hu8_drive",
                    #     "xdc/ibfla_drive": "ao_ibfla_drive",
                    #     "xdc/ibflb_drive": "ao_ibflb_drive",
                    #     "xdc/ibfua_drive": "ao_ibfua_drive",
                    #     "xdc/ibfub_drive": "ao_ibfub_drive",
                    #     "xdc/ibil_drive": "ao_ibil_drive",
                    #     "xdc/ip_e_iperr1": "ip_e_iperr1",
                    #     "xdc/ip_e_iperr2": "ip_e_iperr2",
                    #     "xdc/ip_s_iptarg": "ip_s_iptarg",
                    #     "xdc/ip_s_irestat": "ip_s_irestat",
                    #     "xdc/ip_s_time": "ip_s_time",
                    #     "xdc/ip_t_ipref": "ip_t_ipref",
                    #     "xdc/loopback_out": "do_loopback_out",
                    #     "xdc/lvcc05": "ai_raw_lvcc05",
                    #     "xdc/nbiss_enable": "do_nbiss_enable",
                    #     "xdc/nbi_ss_i": "ai_raw_nbi_ss_i",
                    #     "xdc/nbiss_notch": "do_nbiss_notch",
                    #     "xdc/nbi_ss_v": "ai_raw_nbi_ss_v",
                    #     "xdc/nbisw_enable": "do_nbisw_enable",
                    #     "xdc/nbi_sw_i": "ai_raw_nbi_sw_i",
                    #     "xdc/nbi_sw_v": "ai_raw_nbi_sw_v",
                    #     "xdc/obr01": "ai_raw_obr01",
                    #     "xdc/obr02": "ai_raw_obr02",
                    #     "xdc/obr03": "ai_raw_obr03",
                    #     "xdc/obr04": "ai_raw_obr04",
                    #     "xdc/obr05": "ai_raw_obr05",
                    #     "xdc/obr06": "ai_raw_obr06",
                    #     "xdc/obr07": "ai_raw_obr07",
                    #     "xdc/obr08": "ai_raw_obr08",
                    #     "xdc/obr09": "ai_raw_obr09",
                    #     "xdc/obr10": "ai_raw_obr10",
                    #     "xdc/obr11": "ai_raw_obr11",
                    #     "xdc/obr12": "ai_raw_obr12",
                    #     "xdc/obr13": "ai_raw_obr13",
                    #     "xdc/obr14": "ai_raw_obr14",
                    #     "xdc/obr15": "ai_raw_obr15",
                    #     "xdc/obr16": "ai_raw_obr16",
                    #     "xdc/obr17": "ai_raw_obr17",
                    #     "xdc/obr18": "ai_raw_obr18",
                    #     "xdc/obr19": "ai_raw_obr19",
                    #     "xdc/obv01": "ai_raw_obv01",
                    #     "xdc/obv02": "ai_raw_obv02",
                    #     "xdc/obv03": "ai_raw_obv03",
                    #     "xdc/obv04": "ai_raw_obv04",
                    #     "xdc/obv05": "ai_raw_obv05",
                    #     "xdc/obv06": "ai_raw_obv06",
                    #     "xdc/obv07": "ai_raw_obv07",
                    #     "xdc/obv08": "ai_raw_obv08",
                    #     "xdc/obv09": "ai_raw_obv09",
                    #     "xdc/obv10": "ai_raw_obv10",
                    #     "xdc/obv11": "ai_raw_obv11",
                    #     "xdc/obv12": "ai_raw_obv12",
                    #     "xdc/obv13": "ai_raw_obv13",
                    #     "xdc/obv14": "ai_raw_obv14",
                    #     "xdc/obv15": "ai_raw_obv15",
                    #     "xdc/obv16": "ai_raw_obv16",
                    #     "xdc/obv17": "ai_raw_obv17",
                    #     "xdc/obv18": "ai_raw_obv18",
                    #     "xdc/obv19": "ai_raw_obv19",
                    #     "xdc/p1_drive": "ao_p1_drive",
                    #     "xdc/p2_drive": "ao_p2_drive",
                    #     "xdc/p4_drive": "ao_p4_drive",
                    #     "xdc/p5_drive": "ao_p5_drive",
                    #     "xdc/pf_e_p1err": "pf_e_p1err",
                    #     "xdc/pf_e_p2err": "pf_e_p2err",
                    #     "xdc/pf_e_p54derr": "pf_e_p54derr",
                    #     "xdc/pf_e_p54serr": "pf_e_p54serr",
                    #     "xdc/pf_f_dtest13": "pf_f_dtest13",
                    #     "xdc/pf_f_dtest14": "pf_f_dtest14",
                    #     "xdc/pf_f_dtest15": "pf_f_dtest15",
                    #     "xdc/pf_f_dtest16": "pf_f_dtest16",
                    #     "xdc/pf_f_p1": "pf_f_p1",
                    #     "xdc/pf_f_p2": "pf_f_p2",
                    #     "xdc/pf_f_p4": "pf_f_p4",
                    #     "xdc/pf_f_p5": "pf_f_p5",
                    #     "xdc/pf_p_p1err": "pf_p_p1err",
                    #     "xdc/pf_p_p2err": "pf_p_p2err",
                    #     "xdc/pf_s_i2t": "pf_s_i2t",
                    #     "xdc/pf_s_i2t_ext": "pf_s_i2t_ext",
                    #     "xdc/pf_s_test1": "pf_s_test1",
                    #     "xdc/pf_s_test2": "pf_s_test2",
                    #     "xdc/pf_s_test3": "pf_s_test3",
                    #     "xdc/pf_s_test4": "pf_s_test4",
                    #     "xdc/pf_s_test5": "pf_s_test5",
                    #     "xdc/pf_t_bv": "pf_t_bv",
                    #     "xdc/pf_t_bvgain": "pf_t_bvgain",
                    #     "xdc/pf_t_idiv": "pf_t_idiv",
                    #     "xdc/pf_t_ipgain": "pf_t_ipgain",
                    #     "xdc/pf_t_p1ff": "pf_t_p1ff",
                    #     "xdc/pf_t_p1gain": "pf_t_p1gain",
                    #     "xdc/pf_t_p1max": "pf_t_p1max",
                    #     "xdc/pf_t_p1min": "pf_t_p1min",
                    #     "xdc/pf_t_p1ref": "pf_t_p1ref",
                    #     "xdc/pf_t_p2ff": "pf_t_p2ff",
                    #     "xdc/pf_t_p2ref": "pf_t_p2ref",
                    #     "xdc/pf_t_p54dff": "pf_t_p54dff",
                    #     "xdc/pf_t_p54dref": "pf_t_p54dref",
                    #     "xdc/pf_t_p54sff": "pf_t_p54sff",
                    #     "xdc/pf_t_p54sgain": "pf_t_p54sgain",
                    #     "xdc/pf_t_p54sref": "pf_t_p54sref",
                    #     "xdc/plasma_current": "ai_raw_plasma_current",
                    #     "xdc/rogext_efcc02": "ai_raw_rogext_efcc02",
                    #     "xdc/rogext_efcc05": "ai_raw_rogext_efcc05",
                    #     "xdc/rogext_efps": "ai_raw_rogext_efps",
                    #     "xdc/rogext_p2li": "ai_raw_rogext_p2li",
                    #     "xdc/rogext_p2lo": "ai_raw_rogext_p2lo",
                    #     "xdc/rogext_p2ui": "ai_raw_rogext_p2ui",
                    #     "xdc/rogext_p2uo": "ai_raw_rogext_p2uo",
                    #     "xdc/rogext_p3l": "ai_raw_rogext_p3l",
                    #     "xdc/rogext_p3u": "ai_raw_rogext_p3u",
                    #     "xdc/rogext_p4l": "ai_raw_rogext_p4l",
                    #     "xdc/rogext_p4u": "ai_raw_rogext_p4u",
                    #     "xdc/rogext_p5l": "ai_raw_rogext_p5l",
                    #     "xdc/rogext_p5u": "ai_raw_rogext_p5u",
                    #     "xdc/rogext_p6a": "ai_raw_rogext_p6a",
                    #     "xdc/rogext_p6b": "ai_raw_rogext_p6b",
                    #     "xdc/rogext_sol": "ai_raw_rogext_sol",
                    #     "xdc/rogext_test": "ai_raw_rogext_test",
                    #     "xdc/rog_ip02_1": "ai_raw_rog_ip02_1",
                    #     "xdc/rog_ip02_2": "ai_raw_rog_ip02_2",
                    #     "xdc/rog_ip02_3": "ai_raw_rog_ip02_3",
                    #     "xdc/rog_ip02_4": "ai_raw_rog_ip02_4",
                    #     "xdc/rog_p2l_1": "ai_raw_rog_p2l_1",
                    #     "xdc/rog_p2u_1": "ai_raw_rog_p2u_1",
                    #     "xdc/rog_p3l_2": "ai_raw_rog_p3l_2",
                    #     "xdc/rog_p3u_2": "ai_raw_rog_p3u_2",
                    #     "xdc/rog_p4l_2": "ai_raw_rog_p4l_2",
                    #     "xdc/rog_p4u_2": "ai_raw_rog_p4u_2",
                    #     "xdc/rog_p5l_2": "ai_raw_rog_p5l_2",
                    #     "xdc/rog_p5u_2": "ai_raw_rog_p5u_2",
                    #     "xdc/rog_p6l_2": "ai_raw_rog_p6l_2",
                    #     "xdc/rog_p6u_2": "ai_raw_rog_p6u_2",
                    #     "xdc/r_outer": "ai_raw_r_outer",
                    #     "xdc/rp_drive": "ao_rp_drive",
                    #     "xdc/rp_f_out": "rp_f_out",
                    #     "xdc/rp_t_ref": "rp_t_ref",
                    #     "xdc/shape_e_bverr1": "shape_e_bverr1",
                    #     "xdc/shape_s_camera_used": "shape_s_camera_used",
                    #     "xdc/shape_s_delta_psi": "shape_s_delta_psi",
                    #     "xdc/shape_s_dpsi_dr": "shape_s_dpsi_dr",
                    #     "xdc/shape_s_dpsiref": "shape_s_dpsiref",
                    #     "xdc/shape_s_fluxerr1": "shape_s_fluxerr1",
                    #     "xdc/shape_s_fluxerr2": "shape_s_fluxerr2",
                    #     "xdc/shape_s_fluxerr3": "shape_s_fluxerr3",
                    #     "xdc/shape_s_fluxerr4": "shape_s_fluxerr4",
                    #     "xdc/shape_s_fluxerr5": "shape_s_fluxerr5",
                    #     "xdc/shape_t_drref": "shape_t_drref",
                    #     "xdc/system_f_dmv_t": "system_f_dmv_t",
                    #     "xdc/system_f_lbout": "system_f_lbout",
                    #     "xdc/system_f_ss_en": "system_f_ss_en",
                    #     "xdc/system_f_ss_notch": "system_f_ss_notch",
                    #     "xdc/system_f_sw_en": "system_f_sw_en",
                    #     "xdc/system_f_wdogen": "system_f_wdogen",
                    #     "xdc/system_f_wdogsq": "system_f_wdogsq",
                    #     "xdc/system_s_average": "system_s_average",
                    #     "xdc/system_s_cycle": "system_s_cycle",
                    #     "xdc/system_s_diag1": "system_s_diag1",
                    #     "xdc/system_s_diag2": "system_s_diag2",
                    #     "xdc/system_s_diag3": "system_s_diag3",
                    #     "xdc/system_s_errors": "system_s_errors",
                    #     "xdc/system_s_overrun": "system_s_overrun",
                    #     "xdc/system_s_polling": "system_s_polling",
                    #     "xdc/system_s_ss_pnbi": "system_s_ss_pnbi",
                    #     "xdc/system_s_ss_pshine": "system_s_ss_pshine",
                    #     "xdc/system_s_sstd_deep": "system_s_sstd_deep",
                    #     "xdc/system_s_sst_deep": "system_s_sst_deep",
                    #     "xdc/system_s_sstd_surf": "system_s_sstd_surf",
                    #     "xdc/system_s_sst_slice1": "system_s_sst_slice1",
                    #     "xdc/system_s_sst_surf": "system_s_sst_surf",
                    #     "xdc/system_s_sw_pnbi": "system_s_sw_pnbi",
                    #     "xdc/system_s_sw_pshine": "system_s_sw_pshine",
                    #     "xdc/system_s_swtd_deep": "system_s_swtd_deep",
                    #     "xdc/system_s_swt_deep": "system_s_swt_deep",
                    #     "xdc/system_s_swtd_surf": "system_s_swtd_surf",
                    #     "xdc/system_s_swt_slice1": "system_s_swt_slice1",
                    #     "xdc/system_s_swt_surf": "system_s_swt_surf",
                    #     "xdc/tc11_drive": "ao_tc11_drive",
                    #     "xdc/tc5a_drive": "ao_tc5a_drive",
                    #     "xdc/tc5b_drive": "ao_tc5b_drive",
                    #     "xdc/testch1": "ai_raw_testch1",
                    #     "xdc/testch2": "ai_raw_testch2",
                    #     "xdc/testch3": "ai_raw_testch3",
                    #     "xdc/tf_current": "ai_raw_tf_current",
                    #     "xdc/wdog_enable": "do_wdog_enable",
                    #     "xdc/wdog_square": "do_wdog_square",
                    #     "xdc/z_d_gain": "z_d_gain",
                    #     "xdc/z_drive": "z_drive",
                    #     "xdc/z_f_zdg": "z_f_zdg",
                    #     "xdc/z_f_zout": "z_f_zout",
                    #     "xdc/z_s_pterm": "z_s_pterm",
                    #     "xdc/z_s_zip": "z_s_zip",
                    #     "xdc/z_s_zipref": "z_s_zipref",
                    #     "xdc/z_t_zdgain": "z_t_zdgain",
                    #     "xdc/z_t_zpgain": "z_t_zpgain",
                    #     "xdc/z_t_zref": "z_t_zref",
                    #     "xdc/z_t_ztest": "z_t_ztest",
                    # })),

                    MapDict(StandardiseSignalDataset("xdc")),
                    MergeDatasets(),
                    TensoriseChannels(
                        "ai_cpu1_ccbv",
                        dim_name="ai_ccbv_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_flcc",
                        dim_name="ai_flcc_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_incon",
                        dim_name="ai_incon_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_lhorw",
                        dim_name="ai_lhorw_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_mid",
                        dim_name="ai_mid_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_obr",
                        dim_name="ai_obr_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_obv",
                        dim_name="ai_obv_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_ring",
                        dim_name="ai_ring_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_rodgr",
                        dim_name="ai_rodgr_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_uhorw",
                        dim_name="ai_uhorw_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_cpu1_vertw",
                        dim_name="ai_vertw_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_raw_ccbv", dim_name="ai_ccbv", assign_coords=False
                    ),
                    TensoriseChannels(
                        "ai_raw_flcc",
                        dim_name="ai_flcc_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "ai_raw_obv", dim_name="ai_obv_channel", assign_coords=False
                    ),
                    TensoriseChannels(
                        "ai_raw_obr", dim_name="ai_obr_channel", assign_coords=False
                    ),
                    TensoriseChannels(
                        "equil_s_seg",
                        regex=r"equil_s_seg(\d+)$",
                        dim_name="equil_seg_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "equil_s_seg_at",
                        regex=r"equil_s_seg(\d+)at$",
                        dim_name="equil_seg_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "equil_s_seg_rt",
                        regex=r"equil_s_seg(\d+)rt$",
                        dim_name="equil_seg_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "equil_s_seg_zt",
                        regex=r"equil_s_seg(\d+)zt$",
                        dim_name="equil_seg_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "equil_s_segb",
                        dim_name="equil_seg_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "equil_t_seg",
                        regex=r"equil_t_seg(\d+)$",
                        dim_name="equil_seg_channel",
                        assign_coords=False,
                    ),
                    TensoriseChannels(
                        "equil_t_seg_u",
                        regex=r"equil_t_seg(\d+)u$",
                        dim_name="equil_seg_channel",
                        assign_coords=False,
                    ),
                    RenameVariables({
                        'ai_cpu1_botcol': 'ai_botcol',                     
                        'ai_cpu1_camera_ok': 'ai_camera_ok',      
                        'ai_cpu1_ccbv': 'ai_ccbv',       
                        'ai_cpu1_co2': 'ai_co2',        
                        'ai_cpu1_endcrown_l': 'ai_endcrown_l', 
                        'ai_cpu1_endcrown_u': 'ai_endcrown_u', 
                        'ai_cpu1_flcc': 'ai_flcc',       
                        'ai_cpu1_flp2l1': 'ai_flp2l1',     
                        'ai_cpu1_flp2l2': 'ai_flp2l2',    
                        'ai_cpu1_flp2l3': 'ai_flp2l3',    
                        'ai_cpu1_flp2l4': 'ai_flp2l4',    
                        'ai_cpu1_flp2u1': 'ai_flp2u1',    
                        'ai_cpu1_flp2u2': 'ai_flp2u2',    
                        'ai_cpu1_flp2u3': 'ai_flp2u3',         
                        'ai_cpu1_flp2u4': 'ai_flp2u4',        
                        'ai_cpu1_flp3l1': 'ai_flp3l1',         
                        'ai_cpu1_flp3l4': 'ai_flp3l4',      
                        'ai_cpu1_flp3u1': 'ai_flp3u1',         
                        'ai_cpu1_flp3u4': 'ai_flp3u4',        
                        'ai_cpu1_flp4l1': 'ai_flp4l1',         
                        'ai_cpu1_flp4l4': 'ai_flp4l4',      
                        'ai_cpu1_flp4u1': 'ai_flp4u1',         
                        'ai_cpu1_flp4u4': 'ai_flp4u4',         
                        'ai_cpu1_flp5l1': 'ai_flp5l1',         
                        'ai_cpu1_flp5l4': 'ai_flp5l4',         
                        'ai_cpu1_flp5u1': 'ai_flp5u1',      
                        'ai_cpu1_flp5u4': 'ai_flp5u4',         
                        'ai_cpu1_flp6l1': 'ai_flp6l1',      
                        'ai_cpu1_flp6u1': 'ai_flp6u1',         
                        'ai_cpu1_hene': 'ai_hene',        
                        'ai_cpu1_incon': 'ai_incon',          
                        'ai_cpu1_lhorw': 'ai_lhorw',          
                        'ai_cpu1_lvcc05': 'ai_lvcc05',         
                        'ai_cpu1_mid': 'ai_mid',            
                        'ai_cpu1_nbi_ss_i': 'ai_nbi_ss_i',       
                        'ai_cpu1_nbi_ss_v': 'ai_nbi_ss_v',       
                        'ai_cpu1_nbi_sw_i': 'ai_nbi_sw_i',       
                        'ai_cpu1_nbi_sw_v': 'ai_nbi_sw_v',       
                        'ai_cpu1_obr': 'ai_obr',            
                        'ai_cpu1_obv': 'ai_obv',            
                        'ai_cpu1_p2l_case': 'ai_p2l_case',      
                        'ai_cpu1_p2larm1': 'ai_p2larm1',        
                        'ai_cpu1_p2larm2': 'ai_p2larm2',     
                        'ai_cpu1_p2larm3': 'ai_p2larm3',       
                        'ai_cpu1_p2ldivpl1': 'ai_p2ldivpl1',     
                        'ai_cpu1_p2ldivpl2': 'ai_p2ldivpl2',     
                        'ai_cpu1_p2u_case': 'ai_p2u_case',      
                        'ai_cpu1_p2uarm1': 'ai_p2uarm1',                                                  
                        'ai_cpu1_p2uarm2': 'ai_p2uarm2',        
                        'ai_cpu1_p2uarm3': 'ai_p2uarm3',                                                        
                        'ai_cpu1_p2udivpl1': 'ai_p2udivpl1',                                                      
                        'ai_cpu1_p2udivpl2': 'ai_p2udivpl2',                                                                    
                        'ai_cpu1_p3l_case': 'ai_p3l_case',                                                       
                        'ai_cpu1_p3u_case': 'ai_p3u_case',                                                                     
                        'ai_cpu1_p4l_case': 'ai_p4l_case',                                                                     
                        'ai_cpu1_p4u_case': 'ai_p4u_case',                                                                     
                        'ai_cpu1_p5l_case': 'ai_p5l_case',                                                                     
                        'ai_cpu1_p5u_case': 'ai_p5u_case',                                                                     
                        'ai_cpu1_p6l_case': 'ai_p6l_case',                                                                     
                        'ai_cpu1_p6u_case': 'ai_p6u_case',                                                                     
                        'ai_cpu1_plasma_current': 'ai_plasma_current',                                                               
                        'ai_cpu1_r_outer': 'ai_r_outer',                                                                      
                        'ai_cpu1_r_outer_ground': 'ai_r_outer_ground',                                                               
                        'ai_cpu1_r_outer_inv': 'ai_r_outer_inv',                                                                  
                        'ai_cpu1_r_outer_signal': 'ai_r_outer_signal',                                                               
                        'ai_cpu1_ring': 'ai_ring',                                                                         
                        'ai_cpu1_rodgr': 'ai_rodgr',                                                                        
                        'ai_cpu1_rog_ip02_1': 'ai_rog_ip02_1',                                                                        
                        'ai_cpu1_rog_ip02_2': 'ai_rog_ip02_2',                                                                   
                        'ai_cpu1_rog_ip02_3': 'ai_rog_ip02_3',                                                                        
                        'ai_cpu1_rog_ip02_4': 'ai_rog_ip02_4',                                                                        
                        'ai_cpu1_rog_p2l_1': 'ai_rog_p2l_1',                                                                         
                        'ai_cpu1_rog_p2u_1': 'ai_rog_p2u_1',                                                                         
                        'ai_cpu1_rog_p3l_2': 'ai_rog_p3l_2',                                                                           
                        'ai_cpu1_rog_p3u_2': 'ai_rog_p3u_2',                                                                         
                        'ai_cpu1_rog_p4l_2': 'ai_rog_p4l_2',                                                                           
                        'ai_cpu1_rog_p4u_2': 'ai_rog_p4u_2',                                                                             
                        'ai_cpu1_rog_p5l_2': 'ai_rog_p5l_2',                                                                           
                        'ai_cpu1_rog_p5u_2': 'ai_rog_p5u_2',                                                                               
                        'ai_cpu1_rog_p6l_2': 'ai_rog_p6l_2',                                                                             
                        'ai_cpu1_rog_p6u_2': 'ai_rog_p6u_2',                                                                                  
                        'ai_cpu1_rogext_efcc02': 'ai_rogext_efcc02',                                                                           
                        'ai_cpu1_rogext_efcc05': 'ai_rogext_efcc05',                                                                               
                        'ai_cpu1_rogext_efps': 'ai_rogext_efps',                                                                                
                        'ai_cpu1_rogext_p2li': 'ai_rogext_p2li',                                                                                 
                        'ai_cpu1_rogext_p2lo': 'ai_rogext_p2lo',                                                                                   
                        'ai_cpu1_rogext_p2ui': 'ai_rogext_p2ui',                                                                                 
                        'ai_cpu1_rogext_p2uo': 'ai_rogext_p2uo',                                                                                   
                        'ai_cpu1_rogext_p3l': 'ai_rogext_p3l',                                                                                          
                        'ai_cpu1_rogext_p3u': 'ai_rogext_p3u',                                                                                    
                        'ai_cpu1_rogext_p4l': 'ai_rogext_p4l',                                                                                          
                        'ai_cpu1_rogext_p4u': 'ai_rogext_p4u',                                                                                               
                        'ai_cpu1_rogext_p5l': 'ai_rogext_p5l',                                                                                          
                        'ai_cpu1_rogext_p5u': 'ai_rogext_p5u',                                                                                               
                        'ai_cpu1_rogext_p6a': 'ai_rogext_p6a',                                                                                               
                        'ai_cpu1_rogext_p6b': 'ai_rogext_p6b',                                                                                               
                        'ai_cpu1_rogext_sol': 'ai_rogext_sol',                                                                                               
                        'ai_cpu1_rogext_test': 'ai_rogext_test',                                                                                              
                        'ai_cpu1_testch1': 'ai_testch1',                                                                                                  
                        'ai_cpu1_testch2': 'ai_testch2',                                                                                                  
                        'ai_cpu1_testch3': 'ai_testch3',                                                                                                  
                        'ai_cpu1_tf_current': 'ai_tf_current',                                                                                               
                        'ai_cpu1_topcol': 'ai_topcol',                                                                                                           
                        'ai_cpu1_uhorw': 'ai_uhorw',                                                                                                    
                        'ai_cpu1_vertw': 'ai_vertw',                                                                                                            
                        'di_cpu1_cam_ok': 'di_cam_ok',                                                                                                           
                        'di_cpu1_dt0_trigger': 'di_dt0_trigger',                                                                                                      
                        'di_cpu1_loopback_in': 'di_loopback_in',                                                                                                           
                        'di_cpu1_magnetics_ok': 'di_magnetics_ok',                                                                                                     
                        'di_cpu1_nbi_ss_on': 'di_nbi_ss_on',                                                                                                             
                        'di_cpu1_nbi_sw_on': 'di_nbi_sw_on',                                                                                                              
                        'di_cpu1_ntm_kick': 'di_ntm_kick',                                                                                                              
                        'di_cpu1_power_on': 'di_power_on',                                                                                                               
                        'di_cpu1_watchdog_ok': 'di_watchdog_ok',
                    }),
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
                    TensoriseChannels("hcam_l", regex=r"hcam_l_(\d+)"),
                    TensoriseChannels("hcam_u", regex=r"hcam_u_(\d+)"),
                    TensoriseChannels("tcam", regex=r"tcam_(\d+)"),
                    TensoriseChannels("hcam_l", regex=r"hcaml#(\d+)"),
                    TensoriseChannels("hcam_u", regex=r"hcamu#(\d+)"),
                    TensoriseChannels("hpzr", regex=r"hpzr_(\d+)"),
                    TensoriseChannels("v_ste29", regex=r"v_ste29_(\d+)"),
                    TensoriseChannels("v_ste36", regex=r"v_ste36_(\d+)"),
                    TransformUnits(),
                    AddXSXCameraParams("hcam_l", "parameters/xsx_camera_l.csv"),
                    AddXSXCameraParams("hcam_u", "parameters/xsx_camera_u.csv"),
                    AddXSXCameraParams("tcam", "parameters/xsx_camera_t.csv"),
                ]
            ),
        }

