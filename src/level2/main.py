import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pint  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr

from src.core.config import ReaderConfig, load_config
from src.core.load import (
    BaseLoader,
    loader_registry,
)
from src.core.log import logger
from src.core.model import Mapping, load_model
from src.core.upload import UploadS3
from src.core.workflow_manager import WorkflowManager
from src.core.writer import dataset_writer_registry
from src.level2.reader import DatasetReader


def running_mean(x, N=300):
    cumsum = np.cumsum(np.insert(x, 0, 0))
    return (cumsum[N:] - cumsum[:-N]) / float(N)


def trim_ip_range(dataset: xr.Dataset, delta_time: float) -> xr.Dataset:
    NN = 300
    NN_adapt = int(NN * delta_time / (2e-4))

    ip_ = dataset.values / 1000
    rm = running_mean(ip_, NN_adapt)

    sign_ip = rm[abs(rm) > 30].mean()
    sign_ip /= abs(sign_ip)
    ip_ *= sign_ip
    rm *= sign_ip

    count_i = 0
    while abs(rm[count_i]) < 15:
        count_i += 1

    count_o = len(rm) - 1
    while abs(rm[count_o]) < 15:
        count_o -= 1

    dataset = dataset[count_i : count_o + NN_adapt]
    return dataset


def check_plasma_current(plasma_current: xr.DataArray, tdelta: float) -> bool:
    current_check = np.sum(abs(plasma_current.values) > 200 * 1000) > 250 * tdelta / (
        2e-4
    )
    return current_check


def load_mapping_file(mapping_file: str):
    mapping_file = Path(mapping_file)
    if not mapping_file.exists():
        logger.error(f'No mapping file exists called "{mapping_file}"')
        sys.exit(-1)

    mapping = load_model(mapping_file)
    return mapping


def load_config_file(config_file: str):
    mapping_file = Path(config_file)
    if not mapping_file.exists():
        logger.error(f'No mapping file exists called "{mapping_file}"')
        sys.exit(-1)

    mapping = load_config(mapping_file)
    return mapping


def create_uuid(oid_name: str) -> str:
    return str(uuid.uuid5(uuid.NAMESPACE_OID, oid_name))


def get_default_loader(config: ReaderConfig) -> BaseLoader:
    loader_type = config.type
    loader_params = config.options
    loader = loader_registry.create(loader_type, **loader_params)
    return loader


def set_mapping_time_bounds(
    mapping: Mapping, shot: int, tdelta: float, loader: BaseLoader
):
    if mapping.plasma_current is None:
        return

    dataset_name, signal_name = mapping.plasma_current.split("/")
    reader = DatasetReader(mapping, loader)
    plasma_current = reader.read_profile(shot, dataset_name, signal_name)

    if plasma_current is None:
        raise RuntimeError("Cannot load Plasma Current")

    if not check_plasma_current(plasma_current, tdelta):
        raise RuntimeError(f"No Plasma Current for shot {shot}")

    plasma_current = trim_ip_range(plasma_current, tdelta)

    if mapping.global_interpolate.tmin is None:
        tmin = float(plasma_current.time.values.min())
        mapping.global_interpolate.tmin = tmin

    if mapping.global_interpolate.tmax is None:
        tmax = float(plasma_current.time.values.max())
        mapping.global_interpolate.tmax = tmax

def consolidate_flux_loops(ds: xr.Dataset) -> xr.Dataset:
    r_data = []
    z_data = []
    r_coords = []
    z_coords = []
    r_attrs = {}
    z_attrs = {}
    
    for var in ds.data_vars:
        if var.startswith('flux_loop_') and var.endswith('_r'):
            base = var[:-2]  # remove '_r'
            geom_var = f"{base}_geometry_channel"
            if geom_var in ds:
                r_data.extend(ds[var].values)
                r_coords.extend(ds[geom_var].values)
                r_attrs.update(ds[var].attrs)
                
        elif var.startswith('flux_loop_') and var.endswith('_z'):
            base = var[:-2]  # remove '_z'
            geom_var = f"{base}_geometry_channel"
            if geom_var in ds:
                z_data.extend(ds[var].values)
                z_coords.extend(ds[geom_var].values)
                z_attrs.update(ds[var].attrs)
    
    result_arrays = {}
    if r_data:
        r_attrs['description'] = "Major radius of the flux loops"
        result_arrays['flux_loop_r'] = xr.DataArray(
            data=r_data,
            dims="flux_loop_geometry_channel",
            coords={"flux_loop_geometry_channel": r_coords},
            attrs=r_attrs, 
            name="flux_loop_r"
        )
    if z_data:
        z_attrs['description'] = "Vertical position of the flux loops"
        result_arrays['flux_loop_z'] = xr.DataArray(
            data=z_data,
            dims="flux_loop_geometry_channel",
            coords={"flux_loop_geometry_channel": z_coords},
            attrs=z_attrs, 
            name="flux_loop_z"
        )
    
    result_ds = ds.copy()
    for name, array in result_arrays.items():
        result_ds[name] = array
    
    vars_to_remove = []
    coords_to_remove = []
    for var in ds.data_vars:
        if var.startswith('flux_loop_') and (var.endswith('_r') or var.endswith('_z')):
            vars_to_remove.append(var)
    for coord in ds.coords:
        if coord.startswith('flux_loop_') and coord.endswith('_geometry_channel'):
            coords_to_remove.append(coord)
    
    result_ds = result_ds.drop_vars(vars_to_remove)
    result_ds = result_ds.drop_vars(coords_to_remove)
    
    return result_ds


def process_shot(shot: int, **kwargs):
    args = argparse.Namespace(**kwargs)
    if args.verbose:
        logger.setLevel("DEBUG")

    tdelta = args.dt
    dataset_names = args.include_datasets

    mapping_file = Path(args.mapping_file)
    if mapping_file.is_dir():
        mapping_file = mapping_file / f"{shot}.yml"

    mapping = load_mapping_file(mapping_file)

    config = load_config_file(args.config_file)
    if args.output_path is not None:
        config.writer.options["output_path"] = args.output_path

    writer = dataset_writer_registry.create(config.writer.type, **config.writer.options)

    file_name = f"{shot}.{writer.file_extension}"
    local_file = config.writer.options["output_path"] / Path(file_name)

    loader = get_default_loader(config.readers[mapping.default_loader])
    set_mapping_time_bounds(mapping, shot, tdelta, loader)

    for group_name in mapping.datasets.keys():
        if len(dataset_names) == 0 or group_name in dataset_names:
            if group_name not in args.exclude_datasets:
                logger.info(
                    f"Processing {group_name} for shot {shot} from {mapping.facility}"
                )

                reader = DatasetReader(mapping, loader)
                dataset = reader.read_dataset(shot, group_name)
                if len(dataset) == 0:
                    continue

                if group_name == "magnetics":
                    dataset = consolidate_flux_loops(dataset)

                logger.info(
                    f"Writing {group_name} for shot {shot} from {mapping.facility}"
                )
                writer.write(file_name, group_name, dataset)

    if config.upload is not None:
        remote_file = f"{config.upload.base_path}/"

        uploader = UploadS3(config.upload)
        uploader.upload(local_file, remote_file)

    logger.info(f"Done shot {shot}!")


def safe_process_shot(shot, *args, **kwargs):
    try:
        process_shot(shot, *args, **kwargs)
    except Exception as e:
        logger.warning(f"Failed to process shot {shot}: {e}")
        logger.debug("Exception information", exc_info=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mapping_file", type=str)
    parser.add_argument("-c", "--config-file", type=str, default="./configs/level2.yml")
    parser.add_argument("--shot", type=int, default=None)
    parser.add_argument("--shot-min", type=int, default=None)
    parser.add_argument("--shot-max", type=int, default=None)
    parser.add_argument("--dt", type=float, default=0.00025)
    parser.add_argument("-i", "--include-datasets", nargs="+", default=[])
    parser.add_argument("-e", "--exclude-datasets", nargs="+", default=[])
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-o", "--output-path", type=str, default=None)
    parser.add_argument("-n", "--n-workers", type=int, default=None)
    args = parser.parse_args()

    if args.verbose:
        logger.setLevel("DEBUG")

    if args.shot is None:
        if args.shot_min is None or args.shot_max is None:
            logger.error(
                "Must provide both a minimum and maximum shot (--shot-min/--shot-max)"
            )
            sys.exit(-1)
        shots = range(args.shot_min, args.shot_max)
    else:
        shots = [args.shot]

    kwargs = vars(args)
    kwargs.pop("shot")
    workflow_manager = WorkflowManager(safe_process_shot)
    workflow_manager.run_workflows(shots, **kwargs)


if __name__ == "__main__":
    main()
