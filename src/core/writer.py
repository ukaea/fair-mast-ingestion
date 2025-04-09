import json
from abc import ABC, abstractmethod
from enum import Enum
from pathlib import Path

import numpy as np
import xarray as xr
import zarr

from src.core.registry import Registry


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
    def write(self, group_name: str, datasets: dict[str, xr.Dataset]):
        raise NotImplementedError(
            f"Base method {self.__qualname__} for {self.__class__.__name__} not implemented."
        )

    def _convert_dict_attrs_to_json(self, dataset: xr.Dataset):
        for var in dataset.data_vars.values():
            for attr_name, item in var.attrs.items():
                if isinstance(item, dict) or isinstance(item, list):
                    var.attrs[attr_name] = json.dumps(item, cls=NumpyEncoder)

        for attr_name, item in dataset.attrs.items():
            if isinstance(item, dict) or isinstance(item, list):
                var.attrs[attr_name] = json.dumps(item, cls=NumpyEncoder)


class ZarrDatasetWriter(DatasetWriter):
    def __init__(
        self, output_path: str, mode: str = "single", zarr_version: int = 2, **kwargs
    ):
        super().__init__(output_path)
        self.version = zarr_version
        self.mode = mode

    @property
    def file_extension(self):
        return "zarr"

    def write(self, file_name: str, group_name: str, dataset: xr.Dataset):
        self._convert_dict_attrs_to_json(dataset)

        if self.mode == "single":
            self._write_single_zarr(file_name, group_name, dataset)
        else:
            self._write_multi_zarr(file_name, group_name, dataset)

    def _write_single_zarr(self, file_name: str, name: str, dataset: xr.Dataset):
        file_name = self.output_path / file_name
        dataset.to_zarr(
            file_name,
            group=name,
            mode="w",
            zarr_format=self.version,
            consolidated=True,
        )
        zarr.consolidate_metadata(file_name)

    def _write_multi_zarr(self, file_name: str, name: str, dataset: xr.Dataset):
        file_name = Path(file_name)
        path = self.output_path / f"{file_name.stem}/{name}.zarr"
        path.parent.mkdir(exist_ok=True, parents=True)
        dataset.to_zarr(path, mode="a", zarr_format=self.version, consolidated=True)
        zarr.consolidate_metadata(path)


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

    @property
    def file_extension(self):
        return "nc"

    def write(self, file_name: str, group_name: str, dataset: xr.Dataset):
        self._convert_dict_attrs_to_json(dataset)

        if self.mode == "single":
            self._write_single_netcdf(file_name, group_name, dataset)
        else:
            self._write_multi_netcdf(file_name, group_name, dataset)

    def _write_single_netcdf(self, file_name: str, name: str, dataset: xr.Dataset):
        file_name = self.output_path / file_name
        self.output_path.mkdir(exist_ok=True, parents=True)
        dataset.to_netcdf(file_name, group=name, mode="a", engine="h5netcdf")

    def _write_multi_netcdf(self, file_name: str, name: str, dataset: xr.Dataset):
        path = self.output_path / f"{file_name}/{name}.nc"
        path.parent.mkdir(exist_ok=True, parents=True)
        dataset.to_netcdf(path, mode="a", engine="h5netcdf")


class DatasetWriterNames(str, Enum):
    ZARR = "zarr"
    NETCDF = "netcdf"
    PARQUET = "parquet"


dataset_writer_registry = Registry[DatasetWriter]()
dataset_writer_registry.register(DatasetWriterNames.ZARR, ZarrDatasetWriter)
dataset_writer_registry.register(DatasetWriterNames.NETCDF, NetCDFDatasetWriter)
dataset_writer_registry.register(DatasetWriterNames.PARQUET, ParquetDatasetWriter)
