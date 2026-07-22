import base64
import json
import re
import time
import typing as t
from abc import ABC
from enum import Enum
from functools import lru_cache
from typing import Optional

import fsspec
import numpy as np
import pandas as pd
import xarray as xr
import zarr
import zarr.storage
from pydantic import BaseModel

from src.core.model import Channel
from src.core.registry import Registry
from src.core.utils import harmonise_name

LAST_MAST_SHOT = 30471


class MissingMetadataError(Exception):
    pass


class MissingProfileError(Exception):
    pass


class MissingSourceError(Exception):
    pass

class MissingCoordinateError(Exception):
    pass


class DatasetInfo(BaseModel):
    name: str
    description: str
    quality: str


class SignalInfo(BaseModel):
    name: str
    version: int
    description: str
    quality: str
    dataset: str


class BaseLoader(ABC):
    def load(self, *args, **kwargs) -> xr.Dataset:
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )

    def list_datasets(self, shot: int) -> list[DatasetInfo]:
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )

    def list_signals(self, shot: int) -> list[SignalInfo]:
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )


class SALLoader(BaseLoader):
    def __init__(self, host: str = "https://sal.jetdata.eu") -> None:
        self.host = host

    def load(self, shot_num: int, name: str, channels: Optional[list[Channel]] = None):
        try:
            return self.load_signal(shot_num, name)
        except Exception as e:
            raise MissingProfileError(f"{e}, {type(e)}")

    def load_signal(self, shot_num: int, name: str) -> xr.DataArray:
        uri = f"sal://pulse/{shot_num}/ppf/signal/jetppf/{name}"
        dataset = xr.open_dataset(uri, engine="sal", host=self.host)

        data = dataset["data"]
        data.name = name
        return data


class UDALoader(BaseLoader):
    def __init__(self, include_error: bool = False) -> None:
        self._include_error = include_error

    def list_datasets(self, shot: int):
        import pyuda

        try:
            source_infos = self.get_source_infos(shot)
        except (pyuda.UDAException, pyuda.ServerException) as e:
            raise MissingMetadataError(
                f"Could not load signal metadata for shot {shot}: {e}"
            )

        return source_infos

    def list_signals(self, shot: int):
        import pyuda

        try:
            signal_infos = self.get_signal_infos(shot)
            image_infos = self.get_image_infos(shot)
        except (pyuda.UDAException, pyuda.ServerException) as e:
            raise MissingMetadataError(
                f"Could not load signal metadata for shot {shot}: {e}"
            )

        infos = signal_infos + image_infos
        return infos

    def get_source_infos(self, shot_num: int) -> t.List[DatasetInfo]:
        from mast.mast_client import ListType

        client = self._get_client()
        signals = client.list(ListType.SOURCES, shot=shot_num)
        infos = [
            DatasetInfo(
                name=item.source_alias,
                description=item.description,
                quality=self.lookup_status_code(item.status),
            )
            for item in signals
        ]

        # Special case: in MAST-U, soft x rays were moved from XSX -> ASX then back to XSX, but XSX contained raw data signals.
        # Here we drop XSX if ASX is avilable, otherwise we return ASX
        if shot_num > LAST_MAST_SHOT:
            sources = {info.name: info for info in infos}
            if "asx" in sources:
                sources.pop("xsx")
                infos = sources.values()

        return infos

    def lookup_status_code(self, status):
        """Status code mapping from the numeric representation to the meaning"""
        lookup = {
            -1: "Very Bad",
            0: "Bad",
            1: "Not Checked",
            2: "Checked",
            3: "Validated",
        }
        return lookup[status]

    def get_signal_infos(self, shot_num: int) -> t.List[SignalInfo]:
        client = self._get_client()
        signals = client.list_signals(shot=shot_num)

        infos = [
            SignalInfo(
                name=item.signal_name,
                description=item.description,
                version=item.pass_,
                quality=self.lookup_status_code(item.signal_status),
                dataset=item.source_alias,
            )
            for item in signals
        ]
        return infos

    def get_image_infos(self, shot_num: int) -> t.List[SignalInfo]:
        from mast.mast_client import ListType

        client = self._get_client()

        sources = client.list(ListType.SOURCES, shot_num)
        sources = [source for source in sources if source.type == "Image"]
        infos = [
            SignalInfo(
                name=item.source_alias,
                description=item.description,
                version=item.pass_,
                quality=self.lookup_status_code(item.status),
                dataset=item.source_alias,
            )
            for item in sources
        ]
        return infos

    def _get_client(self):
        import pyuda

        client = pyuda.Client()
        client.set_property("get_meta", True)
        client.set_property("timeout", 10)
        return client

    def load(self, shot_num: int, name: str, channels: Optional[list[str]] = None):
        try:
            if channels is not None:
                dataset = self.load_channels(shot_num, name, channels)
            else:
                dataset = self.load_signal(shot_num, name)
        except Exception as e:
            raise MissingProfileError(f"{e}, {type(e)}")

        return dataset

    def load_channels(self, shot_num: int, name: str, channels: list[Channel]):
        scales = {c.name: c.scale for c in channels}
        channels = [c.name for c in channels]
        
        signals = {}

        # Load channels, skipping an missing channels
        shape, dims, coords = None, None, None
        for channel in channels:
            try:
                signal = self.load_signal(shot_num, channel)
                signals[channel] = signal
                if shape is None:
                    shape = signal.shape
                    dims = list(signal.sizes.keys())
                    coords = signal.coords
            except MissingSourceError:
                continue

        # Edge case: could not load any channels.
        if len(signals) == 0:
            raise MissingSourceError(
                f'Could not load profile {name} for shot "{shot_num}". Could not load any channels!'
            )

        # Fill any missing channels with NaN
        for channel in channels:
            if channel not in signals:
                signals[channel] = xr.DataArray(
                    np.full(shape, np.nan), dims=dims, coords=coords
                )

        assert (
            len(signals) == len(channels)
        ), "Number of channels must match number of signals loaded. Check mapping names for duplicates."

        channel_values, channel_template = self._extract_channel_template(channels)
        channels_dim = xr.DataArray(data=channel_values, dims=["channels"])
        channels_dim.name = "channels"
        if channel_template is not None:
            channels_dim.attrs["name"] = channel_template

        # Sometimes channels have inconsistent binning, harmonise them here.
        first_signal = None
        for name, signal in signals.items():
            if first_signal is None:
                first_signal = signal

            dim_map = {
                dim_name: new_dim_name
                for dim_name, new_dim_name in zip(signal.dims, first_signal.dims)
            }
            signal = signal.rename(dim_map)
            signal = signal.interp_like(first_signal, method="zero")
            signals[name] = signal

        signals = [signals[channel] * scales[channel] for channel in channels]
        signals = xr.concat(signals, dim=channels_dim)
        return signals

    def load_signal(self, shot_num: int, name: str) -> xr.Dataset | xr.DataArray:
        dataset = self._open_dataset(shot_num, name)

        # The uda backend returns an error variable for signals but not for images.
        if "error" in dataset:
            return self._prepare_signal(name, dataset)
        return self._prepare_image(name, dataset)

    def _open_dataset(self, shot_num: int, name: str) -> xr.Dataset:
        max_attempts = 2
        for attempt in range(max_attempts):
            try:
                return xr.open_dataset(f"uda://{name}:{shot_num}", engine="uda")
            except RuntimeError as e:
                # Check for SSL error specifically
                if "SSL_ERROR_SSL" in str(e) and attempt < max_attempts - 1:
                    time.sleep(1)
                    continue
                raise MissingSourceError(
                    f'Could not load profile {name} for shot "{shot_num}". Encountered exception: {e}'
                )

    def _prepare_signal(
        self, uda_name: str, dataset: xr.Dataset
    ) -> xr.Dataset | xr.DataArray:
        dataset = dataset.rename(self._normalize_dimension_names(dataset))

        signal_name = harmonise_name(uda_name)
        if signal_name == "time":
            signal_name = "time_"

        data = dataset["data"]
        data.name = signal_name
        data.attrs["name"] = data.name
        data.attrs["uda_name"] = uda_name

        if not self._include_error:
            return data.squeeze(drop=True)

        error = dataset["error"]
        error.name = f"{signal_name}_error"
        error.attrs = {**data.attrs, "name": error.name}

        return xr.merge([data, error]).squeeze(drop=True)

    def _prepare_image(
        self, uda_name: str, dataset: xr.Dataset
    ) -> xr.Dataset | xr.DataArray:
        data = dataset["data"]
        data.name = uda_name
        data.attrs["name"] = uda_name
        data.attrs["uda_name"] = uda_name

        if self._include_error:
            return data.to_dataset()
        return data

    def _normalize_dimension_names(self, dataset: xr.Dataset) -> dict[t.Hashable, str]:
        """Make the dimension names sensible"""
        count = 0
        names = {}
        empty_names = ["", " ", "-"]

        for dim in dataset.sizes:
            name = str(dim)

            # Create names for unlabelled dims
            if name in empty_names:
                name = f"dim_{count}"
                count += 1

            # Normalize weird names to standard names
            names[dim] = re.sub("[^a-zA-Z0-9_\n\\.]", "", name.lower())

        return names

    @staticmethod
    def _extract_channel_template(
        channels: list[str],
    ) -> tuple[list[str], Optional[str]]:
        if len(channels) < 2:
            return channels, None

        seps = "/_-# "
        # Longest common prefix
        prefix = ""
        for chars in zip(*channels):
            if len(set(chars)) > 1:
                break
            prefix += chars[0]

        # Trim back to the last structural separator to avoid chopping
        # mid-word (e.g. 'xmc/CC/MT/2' -> 'xmc/CC/MT/')
        cut = max(prefix.rfind(s) for s in seps)
        prefix = prefix[: cut + 1]

        stripped = [ch[len(prefix) :] for ch in channels]
        suffix = ""
        for chars in zip(*(s[::-1] for s in stripped)):
            if len(set(chars)) > 1:
                break
            suffix = chars[0] + suffix

        cuts = [suffix.find(s) for s in seps if suffix.find(s) >= 0]
        suffix = suffix[min(cuts) :] if cuts else ""

        if not prefix and not suffix:
            return channels, None

        suffixes = [ch[len(prefix):] for ch in channels]
        if any(not s for s in suffixes):
            return channels, None

        end = -len(suffix) if suffix else None
        values = [s[:end] for s in stripped]

        if any(not v for v in values):
            return channels, None

        return values, f"{prefix}{{channel}}{suffix}"


_uda_loader = UDALoader()

def _fetch_uda_geometry_tree(path: str, shot):
    client = _uda_loader._get_client()
    return client.geometry(path, shot, no_cal=True)

@lru_cache()
def _fetch_uda_geom_metadata(shot) -> dict:
    """Fetch and JSON-decode UDA geometry metadata for a shot."""
    client = _uda_loader._get_client()
    raw = client.get(f"GEOM::getMetaData(file={shot})").jsonify()
    return json.loads(raw)


class Level2UDAGeometryLoader():
            
        def run(self, profile_geometry, profile_name):
            """Load geometry data and return xarray structure."""
            self.stem = profile_geometry.stem
            self.name = profile_name
            self.path = profile_geometry.path
            self.shot = profile_geometry.shot
            self.measurement = profile_geometry.measurement
            self.channel_name = profile_geometry.channel_name
            
            return self._fetch_and_process_geometry()

        def _fetch_and_process_geometry(self):
            """Fetch and process geometry data from UDA."""
            geom_data = _fetch_uda_geometry_tree(self.path, self.shot)
            geom_data_json = json.loads(geom_data.data[self.stem].jsonify())
            all_rows = self._extract_rows(geom_data_json)

            if "b_field_tor_probe_saddle" in self.name:
                all_rows = self._process_saddle(all_rows, geom_data)
            elif "cam" in self.name:
                all_rows = self._process_xray(all_rows, geom_data)
            elif "limiter" in self.name:
                all_rows = self._process_limiter(all_rows, geom_data)
            elif any(substr in self.name for substr in ["p2_inner", "p2_outer", "p3_lower", "p3_upper", 
                                                        "p4_lower", "p4_upper", "p5_lower", "p5_upper", 
                                                        "p6_lower", "p6_upper", "sol"]):
                all_rows = self._process_pf(all_rows, geom_data)

            geom_df = pd.DataFrame(all_rows).dropna(subset=['name']).drop(['name_', 'version'], axis=1, errors='ignore')
            geom_df = self._set_geometry_index(geom_df)

            geom_xarray = self._create_xarray(geom_df)

            uda_metadata = _fetch_uda_geom_metadata(self.shot)
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

        def _process_pf(self, all_rows, geom_data):
            """Process poloidal field coil geometry data."""
            for row in all_rows:
                for key, item in row.items():
                    if isinstance(item, dict) and item.get('_type') == 'numpy.ndarray':
                        row[key] = getattr(geom_data.data[f'{self.stem}/data/geom_elements'], key)
            return all_rows

        def _process_limiter(self, all_rows, geom_data):
            """Process saddle coil geometry data."""
            for row in all_rows:
                for key, item in row.items():
                    if isinstance(item, dict) and item.get('_type') == 'numpy.ndarray':
                        row[key] = getattr(geom_data.data['efit/data'], key)
            return all_rows

        def _create_xarray(self, geom_df):
            data = geom_df[f"{self.measurement}"].to_numpy()

            if "b_field_tor_probe_saddle" in self.name:
                data = np.stack(data)
                dims = [self.channel_name, "coordinate"]
                coords = {self.channel_name: geom_df["name"].values, "coordinate": np.arange(data.shape[1])}

            elif "cam" in self.name:
                data = np.stack(data).squeeze()
                dims = [self.channel_name]
                coord_labels = [f"{self.stem}_cam_{i+1}" for i in range(data.shape[0])]
                coords = {self.channel_name: coord_labels}

            elif "limiter" in self.name:
                data = np.stack(data).squeeze()
                dims = [self.channel_name]
                coord_labels = [f"element_{i+1}" for i in range(data.shape[0])]
                coords = {self.channel_name: coord_labels}

            elif any(substr in self.name for substr in [
                    "p2_inner", "p2_outer", "p3_lower", "p3_upper", 
                    "p4_lower", "p4_upper", "p5_lower", "p5_upper", 
                    "p6_lower", "p6_upper", "sol"
                    ]):
                data = np.stack(data).squeeze()
                dims = [self.channel_name]
                coord_labels = [f"coil_element_{i}" for i in range(data.shape[0])]
                coords = {self.channel_name: coord_labels}

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



class ZarrLoader(BaseLoader):
    def __init__(self, base_path: str, **kwargs) -> None:
        self.base_path = base_path
        self.fs = fsspec.filesystem(**kwargs)

    def load(self, shot_num: int, name: str) -> xr.DataArray:
        source, name = name.split("/", maxsplit=1)
        url = f"{self.base_path}/{shot_num}.zarr/{source}"

        try:
            store = zarr.storage.FsspecStore(path=url, fs=self.fs)
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
