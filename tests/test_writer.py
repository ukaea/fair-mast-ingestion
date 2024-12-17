import pytest
import xarray as xr
import zarr

from src.writer import ZarrDatasetWriter

pyuda_import = pytest.importorskip("pyuda")


@pytest.mark.usefixtures("fake_dataset")
def test_write_signal(tmpdir, fake_dataset):
    file_name = tmpdir / "30420.zarr"
    writer = ZarrDatasetWriter(tmpdir)
    writer.write(file_name.basename, "amc", fake_dataset)

    dataset = xr.open_zarr(file_name, group="amc")
    assert dataset["time"].shape == (10,)


@pytest.mark.usefixtures("fake_dataset")
def test_write_image(tmpdir, fake_dataset):
    file_name = tmpdir / "30420.zarr"
    fake_dataset.attrs["name"] = "rir"

    writer = ZarrDatasetWriter(tmpdir)
    writer.write(file_name.basename, "rir", fake_dataset)

    dataset = xr.open_zarr(file_name, group="rir")
    assert dataset["time"].shape == (10,)
