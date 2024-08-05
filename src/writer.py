import uuid
import h5py
import zarr
import xarray as xr
from pathlib import Path


def get_dataset_uuid(shot: int) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, str(shot)))


class DatasetWriter:

    def __init__(self, shot: int, dir_name: str, file_format: str = 'zarr'):
        self.shot = shot
        self.dir_name = Path(dir_name)
        self.dir_name.mkdir(exist_ok=True, parents=True)
        self.dataset_path = self.dir_name / f"{shot}.{file_format}"

    def write_metadata(self):
        if self.dataset_path.suffix == '.zarr':
            fhandle = zarr.open(self.dataset_path)
        else:
            fhandle = h5py.File(self.dataset_path, mode='a')

        with fhandle as f:
            f.attrs["dataset_uuid"] = get_dataset_uuid(self.shot)
            f.attrs["shot_id"] = self.shot

    def write_dataset(self, dataset: xr.Dataset):
        name = dataset.attrs["name"]
        if self.dataset_path.suffix == '.zarr':
            dataset.to_zarr(self.dataset_path, group=name, consolidated=True, mode="w")
        elif self.dataset_path.suffix == '.nc' or self.dataset_path.suffix == '.h5':
            for var in dataset.data_vars.values():
                var.attrs = self.remove_none_keys(var.attrs)
            dataset.attrs = self.remove_none_keys(dataset.attrs)
            mode = 'a' if self.dataset_path.exists() else 'w'
            dataset.to_netcdf(self.dataset_path, group=name, mode=mode, engine='h5netcdf')

    def consolidate_dataset(self):
        if self.dataset_path.suffix != '.zarr':
            return

        zarr.consolidate_metadata(self.dataset_path)
        with zarr.open(self.dataset_path) as f:
            for source in f.keys():
                zarr.consolidate_metadata(self.dataset_path / source)
                for signal in f[source].keys():
                    zarr.consolidate_metadata(self.dataset_path / source / signal)

    def remove_none_keys(self, attrs: dict):
        remove_keys = []
        for key, value in attrs.items():
            if value is None:
                remove_keys.append(key)

        for key in remove_keys:
            attrs.pop(key)

        return attrs
        

    def get_group_name(self, name: str) -> str:
        name = name.replace("/", "_")
        name = name.replace(" ", "_")
        name = name.replace("(", "")
        name = name.replace(")", "")
        name = name.replace(",", "")

        if name.startswith("_"):
            name = name[1:]

        parts = name.split("_")
        if len(parts) > 1:
            name = parts[0] + "/" + "_".join(parts[1:])

        name = name.lower()
        return name
