import argparse
import logging
import sys
from functools import partial

from dask_mpi import initialize

from src.uploader import UploadConfig
from src.utils import read_shot_file
from src.workflow import LocalIngestionWorkflow, S3IngestionWorkflow, WorkflowManager


def get_shot_list(args):
    """Get the list of shot numbers from the cli arguments"""

    if args.shot_file is not None:
        shot_list = read_shot_file(args.shot_file)
    elif args.shot_min is not None and args.shot_max is not None:
        shot_list = list(range(args.shot_min, args.shot_max + 1))
    elif args.shot is not None:
        shot_list = [args.shot]
    else:
        logging.error("One of --shot, --shot-file or --shot-min/max must be set.")
        sys.exit(-1)

    return shot_list


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="FAIR MAST Ingestor",
        description="Parse the MAST archive and writer to Zarr/NetCDF/HDF files",
    )

    parser.add_argument("dataset_path")
    parser.add_argument("--shot-file", type=str)
    parser.add_argument("--shot", type=int)
    parser.add_argument("--shot-min", type=int)
    parser.add_argument("--shot-max", type=int)
    parser.add_argument("--bucket_path")
    parser.add_argument("--credentials_file", default=".s5cfg.stfc")
    parser.add_argument("--serial", default=False, action="store_true")
    parser.add_argument("--endpoint_url", default="https://s3.echo.stfc.ac.uk")
    parser.add_argument("--upload", default=False, action="store_true")
    parser.add_argument("--metadata_dir", default="data/uda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--signal_names", nargs="*", default=[])
    parser.add_argument("--source_names", nargs="*", default=[])
    parser.add_argument("--file_format", choices=["zarr", "nc", "h5"], default="zarr")
    parser.add_argument("--facility", choices=["MAST", "MASTU"], default="MAST")

    args = parser.parse_args()

    shot_list = get_shot_list(args)

    if not args.serial:
        initialize()

    if args.upload:
        bucket_path = args.bucket_path
        # Bucket path must have trailing slash
        bucket_path = (
            bucket_path + "/" if not bucket_path.endswith("/") else bucket_path
        )

        config = UploadConfig(
            credentials_file=args.credentials_file,
            endpoint_url=args.endpoint_url,
            url=bucket_path,
        )
        workflow_cls = partial(S3IngestionWorkflow, upload_config=config)
    else:
        config = None
        workflow_cls = LocalIngestionWorkflow

    for source in args.source_names:
        logging.info("------------------------")
        logging.info(f"Starting source {source}")

        workflow = workflow_cls(
            args.metadata_dir,
            args.dataset_path,
            force=args.force,
            signal_names=args.signal_names,
            source_names=[source],
            file_format=args.file_format,
            facility=args.facility,
        )

        workflow_manager = WorkflowManager(workflow)
        workflow_manager.run_workflows(shot_list, parallel=not args.serial)
        logging.info(f"Finished source {source}")


if __name__ == "__main__":
    main()
