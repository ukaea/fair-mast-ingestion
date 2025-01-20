import argparse
import sys
import uuid
from pathlib import Path

import numpy as np
import pint  # noqa: F401
import pint_xarray  # noqa: F401
import xarray as xr

from src.core.config import IngestionConfig, ReaderConfig, load_config
from src.core.load import (
    BaseLoader,
    loader_registry,
)
from src.core.log import logger
from src.core.metadata import MetadataWriter
from src.core.model import Mapping, load_model
from src.core.upload import UploadS3
from src.core.workflow_manager import WorkflowManager
from src.core.writer import dataset_writer_registry
from src.level2.reader import DatasetReader

MIN_SHOT = 11695
MAX_SHOT = 30472


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


def create_metadata_writer(
    writer: BaseLoader, config: IngestionConfig
) -> MetadataWriter:
    db_path = Path(config.metadatabase_file).absolute()
    uri = f"sqlite:////{db_path}"
    if config.upload is not None:
        remote_path = f"{config.upload.base_path}/"
    else:
        remote_path = writer.output_path

    metadata_writer = MetadataWriter(uri, remote_path)
    return metadata_writer


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
    tmin = float(plasma_current.time.values.min())
    tmax = float(plasma_current.time.values.max())

    mapping.global_interpolate.tmin = tmin
    mapping.global_interpolate.tmax = tmax


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
    metadata_writer = create_metadata_writer(writer, config)

    file_name = f"{shot}.{writer.file_extension}"
    local_file = config.writer.options["output_path"] / Path(file_name)

    if local_file.exists() and not kwargs.get('force', True):
        logger.info(f'Skipping shot {shot} as it already exists')
        return

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

                writer.write(file_name, group_name, dataset)

                metadata_writer.write(shot, dataset)

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
    parser.add_argument("-f", "--force", action="store_true")
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
