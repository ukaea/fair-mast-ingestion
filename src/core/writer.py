import json
import warnings
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path
from typing import Optional

import numpy as np
import xarray as xr
import zarr
from h5netcdf.attrs import _HIDDEN_ATTRS as _NETCDF_RESERVED_ATTRS
from zarr.storage import FsspecStore

from src.core.registry import Registry

warnings.filterwarnings(
    "ignore",
    message="Consolidated metadata is currently not part in the Zarr format 3 specification.*",
)

class NumpyEncoder(json.JSONEncoder):
    """Special json encoder for numpy types"""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)


class DatasetWriter(ABC):
    def __init__(self, output_path: str, **kwargs):
        self.output_path = Path(output_path)

    @property
    def file_extension(self):
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )

    @abstractmethod
    def write(self, file_name: str, group_name: str, dataset: xr.Dataset):
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )

    def finalize(self, file_name: str):
        #Hook called once after every group of a shot has been written.
        return

    def _convert_dict_attrs_to_json(self, dataset: xr.Dataset):
        for var in dataset.data_vars.values():
            for attr_name, item in var.attrs.items():
                if isinstance(item, dict) or isinstance(item, list):
                    var.attrs[attr_name] = json.dumps(item, cls=NumpyEncoder)

        for attr_name, item in dataset.attrs.items():
            if isinstance(item, dict) or isinstance(item, list):
                var.attrs[attr_name] = json.dumps(item, cls=NumpyEncoder)

    def _convert_fixed_strings_to_vlen(self, dataset: xr.Dataset):
        """Store strings as variable-length UTF-8, which has a Zarr V3 spec."""
        for name, var in dataset.variables.items():
            if var.dtype.kind == "U":
                dataset[name] = var.astype(object)


class ZarrDatasetWriter(DatasetWriter):
    def __init__(
        self,
        output_path: str,
        mode: str = "single",
        storage_options: Optional[dict] = None,
        **kwargs,
    ):
        super().__init__(output_path)
        self.is_s3 = str(output_path).startswith("s3://")
        if self.is_s3:
            # Path() collapses "s3://" to "s3:/", so keep S3 URIs as plain strings.
            self.output_path = str(output_path).rstrip("/")
        self.mode = mode
        self.storage_options = storage_options or {}
        # Shot stores that received at least one group this run, consolidated once in
        # finalize() (single mode only).
        self._pending_consolidation: set[str] = set()

    @property
    def file_extension(self):
        return "zarr"

    def _get_store(self, file_name: str):
        """Resolve a write target: an fsspec-backed zarr store for S3, else a Path. """
        if self.is_s3:
            url = f"{self.output_path}/{file_name}"
            return FsspecStore.from_url(url, storage_options=self.storage_options)
        return self.output_path / file_name

    def write(self, file_name: str, group_name: str, dataset: xr.Dataset):
        self._convert_dict_attrs_to_json(dataset)
        self._convert_fixed_strings_to_vlen(dataset)
        if self.mode == "single":
            self._write_single_zarr(file_name, group_name, dataset)
        else:
            self._write_multi_zarr(file_name, group_name, dataset)

    def _write_single_zarr(self, file_name: str, name: str, dataset: xr.Dataset):
        store = self._get_store(file_name)
        dataset.to_zarr(store, group=name, mode="w", consolidated=False)
        self._pending_consolidation.add(file_name)

    def _write_multi_zarr(self, file_name: str, name: str, dataset: xr.Dataset):
        if self.is_s3:
            store = self._get_store(f"{Path(file_name).stem}/{name}.zarr")
        else:
            store = self.output_path / f"{Path(file_name).stem}/{name}.zarr"
            store.parent.mkdir(exist_ok=True, parents=True)
        dataset.to_zarr(store, mode="a", consolidated=True)

    def finalize(self, file_name: str):
        if file_name not in self._pending_consolidation:
            return
        zarr.consolidate_metadata(self._get_store(file_name))
        self._pending_consolidation.discard(file_name)


class MultiWriter(DatasetWriter):
    """Fan a single write out to several writers (e.g. local NetCDF + S3 zarr).

    Each sub-writer derives its own filename from the shot stem so they never collide
    on extension. The sub-writers remain individually accessible for per-writer upload.
    """

    def __init__(self, writers: list[DatasetWriter]):
        if len(writers) == 0:
            raise ValueError("MultiWriter requires at least one writer.")
        self.writers = writers

    @property
    def file_extension(self):
        return self.writers[0].file_extension

    def _sub_name(self, file_name: str, writer: DatasetWriter) -> str:
        return f"{Path(file_name).stem}.{writer.file_extension}"

    def write(self, file_name: str, group_name: str, dataset: xr.Dataset):
        for writer in self.writers:
            writer.write(self._sub_name(file_name, writer), group_name, dataset)

    def finalize(self, file_name: str):
        for writer in self.writers:
            writer.finalize(self._sub_name(file_name, writer))


class ParquetDatasetWriter(DatasetWriter):
    def __init__(self, output_path: str, **kwargs):
        super().__init__(output_path)

    @property
    def file_extension(self):
        return "parquet"

    def write(self, file_name: str, group_name: str, dataset: xr.Dataset):
        df = dataset.to_dataframe()
        path = self.output_path / f"{Path(file_name).stem}/{group_name}.parquet"
        path.parent.mkdir(exist_ok=True, parents=True)
        df.to_parquet(path)


class NetCDFDatasetWriter(DatasetWriter):
    def __init__(self, output_path: str, mode: str = "single", **kwargs):
        super().__init__(output_path)
        self.mode = mode
        # Shot files created during this run. The first group of a shot overwrites
        # (mode="w"), later groups append (mode="a"), so re-running a shot starts from
        # a clean file instead of appending onto a previous run's groups.
        self._initialised: set[str] = set()

    @property
    def file_extension(self):
        return "nc"

    def _mode_for(self, key: str) -> str:
        mode = "w" if key not in self._initialised else "a"
        self._initialised.add(key)
        return mode

    def _rename_reserved_attrs(self, attrs: dict):
        for key in list(attrs):
            if key in _NETCDF_RESERVED_ATTRS:
                attrs[f"uda_{key}"] = attrs.pop(key)

    def _sanitise_reserved_attrs(self, dataset: xr.Dataset):
        """Rename HDF5-reserved attribute names (e.g. the camera loader's
        CLASS="IMAGE" tag, see load.py) so h5netcdf accepts them; the value is
        preserved as uda_<name>. Zarr output keeps the original names."""
        self._rename_reserved_attrs(dataset.attrs)
        for var in dataset.variables.values():
            self._rename_reserved_attrs(var.attrs)

    def write(self, file_name: str, group_name: str, dataset: xr.Dataset):
        # Work on a copy so renaming reserved attributes for NetCDF does not leak into
        # a sibling zarr writer sharing the same dataset in a MultiWriter.
        dataset = dataset.copy()
        self._convert_dict_attrs_to_json(dataset)
        self._sanitise_reserved_attrs(dataset)
        self._convert_fixed_strings_to_vlen(dataset)

        if self.mode == "single":
            self._write_single_netcdf(file_name, group_name, dataset)
        else:
            self._write_multi_netcdf(file_name, group_name, dataset)

    def _write_single_netcdf(self, file_name: str, name: str, dataset: xr.Dataset):
        path = self.output_path / file_name
        self.output_path.mkdir(exist_ok=True, parents=True)
        dataset.to_netcdf(
            path, group=name, mode=self._mode_for(file_name), engine="h5netcdf"
        )

    def _write_multi_netcdf(self, file_name: str, name: str, dataset: xr.Dataset):
        path = self.output_path / f"{file_name}/{name}.nc"
        path.parent.mkdir(exist_ok=True, parents=True)
        # Each group is its own file, so overwrite it for a clean re-run.
        dataset.to_netcdf(path, mode="w", engine="h5netcdf")


class DatasetWriterNames(str, Enum):
    ZARR = "zarr"
    NETCDF = "netcdf"
    PARQUET = "parquet"


dataset_writer_registry = Registry[DatasetWriter]()
dataset_writer_registry.register(DatasetWriterNames.ZARR, ZarrDatasetWriter)
dataset_writer_registry.register(DatasetWriterNames.NETCDF, NetCDFDatasetWriter)
dataset_writer_registry.register(DatasetWriterNames.PARQUET, ParquetDatasetWriter)
