import argparse
import logging
from functools import partial
from dask_mpi import initialize
from mpi4py import MPI
import lakefs
from src.uploader import UploadConfig
from src.workflow import S3IngestionWorkflow, LocalIngestionWorkflow, WorkflowManager
from src.utils import read_shot_file
import subprocess

def main():

    # Initialize the MPI communicator
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        lakefs.repository("example-repo").branch("ingestion").create(source_reference="main")


    initialize()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="FAIR MAST Ingestor",
        description="Parse the MAST archive and writer to Zarr/NetCDF/HDF files",
    )

    parser.add_argument("dataset_path")
    parser.add_argument("shot_file")
    parser.add_argument("--bucket_path")
    parser.add_argument("--credentials_file", default=".s5cfg.stfc")
    parser.add_argument("--serial", default=False, action='store_true')
    parser.add_argument("--endpoint_url", default="https://s3.echo.stfc.ac.uk")
    parser.add_argument("--upload", default=False, action="store_true")
    parser.add_argument("--metadata_dir", default="data/uda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--signal_names", nargs="*", default=[])
    parser.add_argument("--source_names", nargs="*", default=[])
    parser.add_argument("--file_format", choices=['zarr', 'nc', 'h5'], default='zarr')
    parser.add_argument("--facility", choices=['MAST', 'MASTU'], default='MAST')

    args = parser.parse_args()

    if args.upload:
        bucket_path = args.bucket_path
        # Bucket path must have trailing slash
        bucket_path = bucket_path + '/' if not bucket_path.endswith('/') else bucket_path

        config = UploadConfig(
            credentials_file=args.credentials_file,
            endpoint_url=args.endpoint_url,
            url=bucket_path,
        )
        workflow_cls = partial(S3IngestionWorkflow, upload_config=config)
    else:
        config = None
        workflow_cls = LocalIngestionWorkflow


    shot_list = read_shot_file(args.shot_file)

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
            facility=args.facility
        )

        workflow_manager = WorkflowManager(workflow)
        workflow_manager.run_workflows(shot_list, parallel = not args.serial)
        logging.info(f"Finished source {source}")

    for shot in shot_list:
        file_path = args.dataset_path + f"/{shot}.{args.file_format}"
        command = [
            "lakectl", "fs", "upload",
            f"lakefs://example-repo/ingestion/{shot}.{args.file_format}",
            "-s", str(file_path), "--recursive"
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Command executed successfully.")
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("An error occurred while executing the command.")
            print("Error message:", e.stderr)

        command = [
            "lakectl", "commit",
            f"lakefs://example-repo/ingestion/",
            "-m", f"Commit shot {shot}"
        ]

        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print("Command executed successfully.")
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print("An error occurred while executing the command.")
            print("Error message:", e.stderr)


if __name__ == "__main__":
    main()