# from icechunk import IcechunkStore, StorageConfig
import pandas as pd
import xarray as xr

from src.core.writer import (
    NetCDFDatasetWriter,
    ParquetDatasetWriter,
    ZarrDatasetWriter,
)


def test_zarr_writer_single(tmp_path, sample_dataset):
    file_name = tmp_path / "test.zarr"
    group_name = "summary"

    writer = ZarrDatasetWriter(file_name.parent)
    writer.write(file_name.name, group_name, sample_dataset)

    assert file_name.exists()
    ds = xr.open_dataset(file_name, group=group_name)
    assert ds == sample_dataset


# def test_icechunk_writer(tmp_path, sample_dataset):
#     file_name = tmp_path / 'test.zarr'
#     group_name = "summary"

#     writer = IcechunkDatasetWriter(file_name.parent)
#     writer.write(file_name.name, group_name, sample_dataset)

#     assert file_name.exists()

#     storage_config = StorageConfig.filesystem(str(file_name))
#     store = IcechunkStore.open_existing(storage_config, mode='r')
#     ds = xr.open_zarr(store, group=group_name, consolidated=False)
#     assert ds == sample_dataset


def test_zarr_writer_multi(tmp_path, sample_dataset):
    file_name = tmp_path / "test"
    group_name = "summary"

    writer = ZarrDatasetWriter(file_name.parent, mode="multi")
    writer.write(file_name.name, group_name, sample_dataset)

    output_file = file_name / f"{group_name}.zarr"
    assert output_file.exists()

    ds = xr.open_dataset(output_file)
    assert ds == sample_dataset


def test_netcdf_writer_single(tmp_path, sample_dataset):
    file_name = tmp_path / "test.nc"
    group_name = "summary"

    writer = NetCDFDatasetWriter(file_name.parent)
    writer.write(file_name.name, group_name, sample_dataset)

    assert file_name.exists()
    ds = xr.open_dataset(file_name, group=group_name)
    assert ds == sample_dataset


def test_netcdf_writer_multi(tmp_path, sample_dataset):
    file_name = tmp_path / "test"
    group_name = "summary"

    writer = NetCDFDatasetWriter(file_name.parent, mode="multi")
    writer.write(file_name.name, group_name, sample_dataset)

    output_file = file_name / f"{group_name}.nc"
    assert output_file.exists()

    ds = xr.open_dataset(output_file)
    assert ds == sample_dataset


def test_parquet_writer(tmp_path, sample_dataset):
    file_name = tmp_path / "test"
    group_name = "summary"

    writer = ParquetDatasetWriter(file_name.parent)
    writer.write(file_name.name, group_name, sample_dataset)

    output_file = file_name / f"{group_name}.parquet"
    assert output_file.exists()

    ds = pd.read_parquet(output_file)
    assert (ds == sample_dataset.to_dataframe()).all(axis=None)
