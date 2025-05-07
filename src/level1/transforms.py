import json
import re
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Optional

import base64
import numpy as np
import pyuda
import pandas as pd
import pyarrow.parquet as pq
import xarray as xr
from pint import UnitRegistry
from pint.errors import UndefinedUnitError

from src.core.log import logger
from src.core.utils import read_json_file

UNITS_MAPPING_FILE = "mappings/level1/units.json"


class BaseTransform(ABC):
    @abstractmethod
    def __call__(self, datasets: xr.Dataset) -> xr.Dataset:
        raise NotImplementedError("Method is not implemented.")


class MapDict(BaseTransform):
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


class RenameDimensions(BaseTransform):
    def __init__(self, mapping_file: str, squeeze_dataset: bool = True) -> None:
        self.squeeze_dataset = squeeze_dataset

        with Path(mapping_file).open("r") as handle:
            self.dimension_mapping = json.load(handle)

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        name = dataset.attrs["name"]
        group_name = dataset.attrs["source"]
        name = f"{group_name}/{name}"

        if name in self.dimension_mapping:
            dims = self.dimension_mapping[name]
            name = name.split("/", maxsplit=1)[-1]

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


class DropZeroDimensions(BaseTransform):
    def __call__(self, dataset: xr.Dataset) -> Any:
        for key, coord in dataset.coords.items():
            if (coord.values == 0).all():
                dataset = dataset.drop_vars(key)
        dataset = dataset.compute()
        return dataset


class DropZeroDataset(BaseTransform):
    def __call__(self, dataset: xr.Dataset) -> Any:
        for key, item in dataset.data_vars.items():
            if (item.values == 0).all():
                dataset = dataset.drop_vars(key)
        dataset = dataset.compute()
        return dataset


class DropDatasets(BaseTransform):
    def __init__(self, keys: list[str]) -> None:
        self.keys = keys

    def __call__(self, datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
        for key in self.keys:
            if key in datasets:
                datasets.pop(key)
        return datasets


class DropErrors(BaseTransform):
    def __init__(self, keys: list[str]):
        self.keys = keys

    def __call__(self, datasets: dict[str, xr.Dataset]) -> dict[str, xr.Dataset]:
        for key in self.keys:
            datasets[key] = datasets[key].drop(f"{key}_error")
        return datasets


class DropCoordinates(BaseTransform):
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


class RenameVariables(BaseTransform):
    def __init__(self, mapping_file: str):
        self.mapping = read_json_file(mapping_file)

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        group_name = dataset.attrs["source"]

        if group_name not in self.mapping:
            return dataset

        for key, value in self.mapping[group_name].items():
            if key in dataset.dims:
                dataset = dataset.rename_dims({key: value})
            if key in dataset:
                dataset = dataset.rename_vars({key: value})
        dataset = dataset.compute()
        return dataset


class MergeDatasets(BaseTransform):
    def __call__(self, dataset_dict: dict[str, xr.Dataset]) -> xr.Dataset:
        dataset = xr.merge(dataset_dict.values())
        dataset = dataset.compute()
        dataset.attrs = {}
        return dataset


class InterpolateAxis(BaseTransform):
    def __init__(self, axis_name: str, method: str):
        super().__init__()
        self.axis_name = axis_name
        self.method = method

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        axis_values = dataset[self.axis_name].values
        amin = axis_values.min()
        amax = axis_values.max()
        adelta = axis_values[1] - axis_values[0]
        coords = np.arange(amin, amax, adelta)

        datasets = {}
        for k, v in dataset.data_vars.items():
            if self.axis_name in v.dims:
                v = v.dropna(self.axis_name, how="all")
                datasets[k] = v.interp({self.axis_name: coords}, method=self.method)
            else:
                datasets[k] = v
        dataset = xr.merge(datasets.values())
        dataset.attrs = {}
        return dataset


class TensoriseChannels(BaseTransform):
    def __init__(
        self,
        stem: str,
        regex: Optional[str] = None,
        dim_name: Optional[str] = None,
        assign_coords: bool = True,
    ) -> None:
        self.stem = stem
        self.regex = regex if regex is not None else stem + r"(\d+)"
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


class TransformUnits(BaseTransform):
    def __init__(self):
        with Path(UNITS_MAPPING_FILE).open("r") as handle:
            self.units_map = json.load(handle)
        self.ureg = UnitRegistry()

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        for key, array in dataset.data_vars.items():
            dataset[key] = self._update_units(array)

        for key, array in dataset.coords.items():
            dataset[key] = self._update_units(array)

        dataset = dataset.compute()
        return dataset

    def _update_units(self, array: xr.DataArray):
        units = array.attrs.get("units", "")
        units = units.strip()
        units = self.units_map.get(units, units)
        units = "dimensionless" if units == "" else units
        array.attrs["units"] = units
        units = self._parse_units(array)
        return array

    def _parse_units(self, item: xr.DataArray):
        units = item.attrs["units"]

        try:
            units = self.ureg.parse_units(units)
            units = f"{units:#~}"
        except (ValueError, UndefinedUnitError, AssertionError, TypeError) as e:
            logger.warning(
                f'Issue with converting units "{units}" for signal "{item.name}": {e}'
            )

        return item


class LCFSTransform(BaseTransform):
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


class AddGeometry(BaseTransform):
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
                name = f"{stem}_{field.name}"
                self.geom_data[name].attrs.update(field_metadata)
                self.geom_data[name].attrs.update(arrow_metadata)

        for key in self.geom_data.keys():
            self.geom_data[key].attrs["name"] = key

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        geom_data = self.geom_data.copy()
        dataset = xr.merge(
            [dataset, geom_data], combine_attrs="no_conflicts", join="left"
        )
        dataset = dataset.compute()
        return dataset

import json
import numpy as np
import pandas as pd
import xarray as xr
import pyuda
import base64


class AddGeometryUDA(BaseTransform):
    """
    A transformation class to retrieve and process geometry data from the UDA system.
    PF and saddle coil geometry are stored as arrays and processed differently from other signals.
    """
    PF_NAMES = {'p2_inner_upper', 'p2_inner_lower', 'p2_outer_lower', 'p2_outer_upper',
                'p3_upper', 'p3_lower', 'p4_upper', 'p4_lower', 'p5_upper', 'p5_lower',
                'p6_upper', 'p6_lower', 'sol'}
    SADDLE_NAMES = {"sad_out_l", "sad_out_m", "sad_out_u"}
    XRAY_NAMES = {"hcam_l", "hcam_u", "hcam_third", "vcam_inner", "vcam_outer", "tcam"}

    def __init__(self, stem: str, name: str, path: str, shot: int):
        self.stem = stem
        self.name = name
        self.path = path
        self.shot = shot
        self.client = pyuda.Client()
        self.geom_xarray = self._fetch_and_process_geometry()

    def _fetch_and_process_geometry(self):
        """Fetch and process geometry data from UDA."""
        geom_data = self.client.geometry(self.path, self.shot, no_cal=True)
        geom_data_json = json.loads(geom_data.data[self.stem].jsonify())
        all_rows = self._extract_rows(geom_data_json)

        # Process geometry data based on signal type
        if self.name in self.SADDLE_NAMES:
            all_rows = self._process_saddle(all_rows, geom_data)
        elif self.name in self.PF_NAMES:
            all_rows = self._process_pf(all_rows, geom_data)
        elif self.name in self.XRAY_NAMES:
            all_rows = self._process_xray(all_rows, geom_data)

        geom_df = pd.DataFrame(all_rows).dropna(subset=['name']).drop(['name_', 'name', 'version'], axis=1, errors='ignore')
        geom_df.columns = [f"{self.name}_{col}" for col in geom_df.columns]
        geom_df = self._set_geometry_index(geom_df)

        return self._create_xarray(geom_df, geom_data)

    def _extract_rows(self, node, rows=None, current_row=None):
        """Recursively extract data rows from UDA structure."""
        if rows is None:
            rows = []
        if current_row is None:
            current_row = {}

        if isinstance(node, dict):
            if 'name_' in node:
                if current_row:
                    rows.append(current_row.copy())
                current_row = {'name': node['name_']}
            for key, value in node.items():
                if key == 'children':
                    self._extract_rows(value, rows, current_row)
                elif key not in ['signal_type', 'dimensions', 'units']:
                    current_row[key] = value
            if 'name' in current_row and pd.notna(current_row['name']) and current_row not in rows:
                rows.append(current_row)
        elif isinstance(node, list):
            for item in node:
                self._extract_rows(item, rows, current_row)

        return rows

    def _set_geometry_index(self, geom_df):
        """Set the geometry index for the dataframe."""
        index_name = f"{self.name}_geometry_index"
        geom_df[index_name] = [f"{self.name}{i+1:02}" for i in range(len(geom_df))]
        return geom_df.set_index(index_name)

    def _process_saddle(self, all_rows, geom_data):
        """Process saddle coil geometry data."""
        for row in all_rows:
            for key, item in row.items():
                if isinstance(item, dict) and item.get('_type') == 'numpy.ndarray':
                    row[key] = getattr(geom_data.data[f'{self.stem}/{row["name"]}/data/coilPath'], key)
        return all_rows

    def _process_pf(self, all_rows, geom_data):
        """Process poloidal field coil geometry data."""
        for row in all_rows:
            for key, item in row.items():
                if isinstance(item, dict) and item.get('_type') == 'numpy.ndarray':
                    row[key] = getattr(geom_data.data[f'{self.stem}/data/geom_elements'], key)
        return all_rows

    def _process_xray(self, all_rows, geom_data):
        """Process x-ray geometry data."""
        new_rows = {}

        for row in all_rows:
            for key, item in row.items():
                if key == "impact_parameter":
                    new_rows[key] = getattr(geom_data.data[f'{self.stem}/data/'], key)
                elif isinstance(item, dict) and item.get('_type') == 'numpy.ndarray' and key != "impact_parameter":
                    new_rows[f"origin_{key}"] = getattr(geom_data.data[f'{self.stem}/data/origin'], key)
                    new_rows[f"endpoint_{key}"] = getattr(geom_data.data[f'{self.stem}/data/endpoint'], key)
                else:
                    new_rows[key] = item
        return new_rows

    def _create_xarray(self, geom_df, geom_data):
        """Create an xarray dataset from processed geometry data."""
        if self.name in self.SADDLE_NAMES:
            r_arr = np.stack(geom_df[f"{self.name}_r"].to_numpy())
            z_arr = np.stack(geom_df[f"{self.name}_z"].to_numpy())
            phi_arr = np.stack(geom_df[f"{self.name}_phi"].to_numpy())
            element_dim = np.arange(r_arr.shape[1])
            return xr.Dataset(
                {
                    f"{self.name}_r": ([f"{self.name}_geometry_index", "element"], r_arr),
                    f"{self.name}_z": ([f"{self.name}_geometry_index", "element"], z_arr),
                    f"{self.name}_phi": ([f"{self.name}_geometry_index", "element"], phi_arr),
                },
                coords={f"{self.name}_geometry_index": geom_df.index.to_numpy(), "element": element_dim}
            )
        elif self.name in self.PF_NAMES:
            centre_r_arr = np.stack(geom_df[f"{self.name}_centreR"].to_numpy()).squeeze()
            centre_z_arr = np.stack(geom_df[f"{self.name}_centreZ"].to_numpy()).squeeze()
            dr_arr = np.stack(geom_df[f"{self.name}_dR"].to_numpy()).squeeze()
            dz_arr = np.stack(geom_df[f"{self.name}_dZ"].to_numpy()).squeeze()
            element_dim = np.arange(dr_arr.shape[0])
            return xr.Dataset(
                {
                    f"{self.name}_coil_r": (["element"], centre_r_arr),
                    f"{self.name}_coil_z": (["element"], centre_z_arr),
                    f"{self.name}_coil_dR": (["element"], dr_arr),
                    f"{self.name}_coil_dZ": (["element"], dz_arr),
                },
                coords={"element": element_dim}
            )
        
        # can comment this out if dont mind repeating the scalar values for each channel?
        elif self.name in self.XRAY_NAMES:
            geometry_index = geom_df.index.astype(str)
            coords = {f"{self.name}_geometry_index": geometry_index}
            ds_vars = {
                col: (f"{self.name}_geometry_index", geom_df[col].to_numpy()) 
                for col in geom_df.columns if geom_df[col].nunique(dropna=False) > 1
            }
            ds_vars.update({
                col: geom_df[col].iloc[0] 
                for col in geom_df.columns if geom_df[col].nunique(dropna=False) == 1
            })

            for col in geom_df.columns:
                if col.startswith(f"{self.name}_endpoint_") or col.startswith(f"{self.name}_origin_"):
                    ds_vars[col] = (f"{self.name}_geometry_index", geom_df[col].to_numpy())

            return xr.Dataset(ds_vars, coords=coords)

        return geom_df.to_xarray()

    def _decode_metadata(self, uda_metadata):
        """Decode UDA metadata, converting base64 to numpy arrays."""
        cleaned_metadata = {}
        for key, value in uda_metadata.items():
            if isinstance(value, dict) and value.get('_type') == 'numpy.ndarray':
                decoded_value = base64.b64decode(value['data']['value'])
                cleaned_metadata[key] = np.frombuffer(decoded_value, dtype=np.int64)[0]
            else:
                cleaned_metadata[key] = value
        return cleaned_metadata

    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        """Merge processed geometry data with the existing dataset."""
        return xr.merge([dataset, self.geom_xarray], combine_attrs="no_conflicts", join="left").compute()


class AddToroidalAngle2(BaseTransform):
    def __init__(self, stem: str, var_name: str, phi_2_value: int = 330):
        self.stem = stem
        self.var_name = var_name
        self.phi_2_value = phi_2_value
    
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        if self.var_name not in dataset.dims:
            raise ValueError(f"Dimension '{self.var_name}' not found in dataset.")
        
        if f"{self.stem}_phi" in dataset:
            dataset = dataset.rename({f"{self.stem}_phi": f"{self.stem}_phi_1"})
        
        # Add 'ccbv_phi_2' with values of 330
        phi_2 = np.full(dataset.sizes[self.var_name], self.phi_2_value)
        dataset[f"{self.stem}_phi_2"] = xr.DataArray(phi_2, dims=[self.var_name])
        
        return dataset
class AlignChannels(BaseTransform):
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


class ProcessImage(BaseTransform):
    def __call__(self, dataset: dict[str, xr.Dataset]) -> xr.Dataset:
        dataset: xr.Dataset = list(dataset.values())[0]
        dataset.attrs["units"] = "pixels"
        dataset = dataset.compute()
        return dataset


class ReplaceInvalidValues(BaseTransform):
    def __call__(self, dataset: xr.Dataset) -> xr.Dataset:
        dataset = dataset.where(dataset != -999, np.nan)
        dataset = dataset.compute()
        return dataset
