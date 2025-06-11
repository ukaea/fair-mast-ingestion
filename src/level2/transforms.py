import base64
import json
import warnings
from abc import ABC
from typing import Union

import numpy as np
import pandas as pd
import pyuda
import scipy.signal
import xarray as xr

from src.core.log import logger
from src.core.model import (
    DatasetInfo,
    Dimension,
    FillOptions,
    GlobalInterpolateParams,
    InterpolationParams,
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
        angle_name = f"{signal_name}_angles"
        angles.attrs["name"] = angle_name
        angles.attrs["units"] = "radians"

        spectrum = xr.DataArray(spectrum, coords=dict(frequency=f, time=t))
        spec_name = f"{signal_name}_spectrogram"
        spectrum.attrs["name"] = spec_name

        signal = xr.Dataset({spec_name: spectrum, angle_name: angles})
        return signal

class BackgroundSubtractionTransform(BaseDatasetTransform):
    def __init__(self, start:int, end: int):
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

class AddGeometryUDA(BaseDatasetTransform):

    SADDLE_NAMES = {"b_field_tor_probe_saddle_l_r", "b_field_tor_probe_saddle_m_r", "b_field_tor_probe_saddle_u_r",
                    "b_field_tor_probe_saddle_l_z", "b_field_tor_probe_saddle_m_z", "b_field_tor_probe_saddle_u_z",
                    "b_field_tor_probe_saddle_l_phi", "b_field_tor_probe_saddle_m_phi", "b_field_tor_probe_saddle_u_phi"}
    XRAY_NAMES = {"tangential_cam_origin_r", "tangential_cam_origin_z", "tangential_cam_endpoint_r", "tangential_cam_endpoint_z", "tangential_cam_phi",
                "horizontal_cam_lower_origin_r", "horizontal_cam_lower_origin_z", "horizontal_cam_lower_endpoint_r", "horizontal_cam_lower_endpoint_z", "horizontal_cam_lower_phi",
                "horizontal_cam_upper_origin_r", "horizontal_cam_upper_origin_z", "horizontal_cam_upper_endpoint_r", "horizontal_cam_upper_endpoint_z", "horizontal_cam_upper_phi"}


    def __init__(self, stem: str, name: str, path: str, shot: int, measurement: str, channel_name: str):
        self.stem = stem
        self.name = name
        self.path = path
        self.shot = shot
        self.measurement = measurement
        self.channel_name = channel_name
        self.client = pyuda.Client()
        self.geom_xarray = self._fetch_and_process_geometry()

    def _fetch_and_process_geometry(self):
        """Fetch and process geometry data from UDA."""
        geom_data = self.client.geometry(self.path, self.shot, no_cal=True)
        geom_data_json = json.loads(geom_data.data[self.stem].jsonify())
        all_rows = self._extract_rows(geom_data_json)

        if self.name in self.SADDLE_NAMES:
            all_rows = self._process_saddle(all_rows, geom_data)
        elif self.name in self.XRAY_NAMES:
            all_rows = self._process_xray(all_rows, geom_data)

        geom_df = pd.DataFrame(all_rows).dropna(subset=['name']).drop(['name_', 'version'], axis=1, errors='ignore')
        geom_df = self._set_geometry_index(geom_df)

        geom_xarray = self._create_xarray(geom_df)

        uda_metadata = json.loads(self.client.get(f"GEOM::getMetaData(file={self.shot})").jsonify())
        cleaned_metadata = self._decode_metadata(uda_metadata)
        geom_xarray.attrs.update(cleaned_metadata)

        return geom_xarray

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
        index_name = f"{self.name}_channel"
        geom_df[index_name] = [f"{self.name}{i+1:02}" for i in range(len(geom_df))]
        return geom_df.set_index(index_name)
    
    def _process_saddle(self, all_rows, geom_data):
        """Process saddle coil geometry data."""
        for row in all_rows:
            for key, item in row.items():
                if isinstance(item, dict) and item.get('_type') == 'numpy.ndarray':
                    row[key] = getattr(geom_data.data[f'{self.stem}/{row["name"]}/data/coilPath'], key)
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

    def _create_xarray(self, geom_df):
        data = geom_df[f"{self.measurement}"].to_numpy()

        if self.name in self.SADDLE_NAMES:
            data = np.stack(data)
            dims = [self.channel_name, "coordinate"]
            coords = {self.channel_name: geom_df["name"].values, "coordinate": np.arange(data.shape[1])}

        elif self.name in self.XRAY_NAMES:
            dims = [self.channel_name]
            coords = None 

        else:
            dims = [self.channel_name]
            coords = {self.channel_name: geom_df["name"].values}

        return xr.DataArray(
            name=self.name,
            data=data,
            dims=dims,
            coords=coords
        )

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