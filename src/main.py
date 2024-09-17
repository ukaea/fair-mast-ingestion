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
from pathlib import Path
import shutil


def initialize_lakefs_branch():
    """Initialize the lakeFS ingestion branch if on rank 0."""
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    if rank == 0:
        lakefs.repository("example-repo").branch("ingestion").create(source_reference="main")


def execute_lakectl_command(command, error_message):
    """Helper function to execute lakectl commands with error handling."""
    try:
        result = subprocess.run(command, check=True, capture_output=True, text=True)
        logging.info("Command executed successfully.")
        logging.info("Output: %s", result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        logging.error(error_message)
        logging.error("Error message: %s", e.stderr)
        return False


def upload_shot_to_lakefs(shot, dataset_path, file_format):
    """Upload a specific shot to lakeFS."""
    file_path = f"{dataset_path}/{shot}.{file_format}"
    upload_command = [
        "lakectl", "fs", "upload",
        f"lakefs://example-repo/ingestion/{shot}.{file_format}",
        "-s", str(file_path), "--recursive"
    ]
    if execute_lakectl_command(upload_command, f"Failed to upload shot {shot} to lakeFS"):
        logging.info(f"Uploaded shot {shot} to lakeFS.")


def commit_shot_to_lakefs(shot):
    """Commit the uploaded shot to lakeFS."""
    commit_command = [
        "lakectl", "commit",
        "lakefs://example-repo/ingestion/",
        "-m", f"Commit shot {shot}"
    ]
    if execute_lakectl_command(commit_command, f"Failed to commit shot {shot} to lakeFS"):
        logging.info(f"Committed shot {shot} to lakeFS.")
        

def remove_shot_from_local(shot, dataset_path, file_format):
    """Remove the shot file from the local filesystem."""
    file_path = Path(f"{dataset_path}/{shot}.{file_format}")
    try:
        if file_path.exists():
            shutil.rmtree(file_path)
            logging.info(f"Removed local file: {file_path}")
        else:
            logging.warning(f"File {file_path} not found.")
    except Exception as e:
        logging.error(f"Failed to remove file {file_path}: {e}")

def merge_branch():
        logging.info("Merging branch to main...")
        main_branch = lakefs.repository("example-repo").branch("main")
        ingestion_branch = lakefs.repository("example-repo").branch("ingestion")
        try:
            res = ingestion_branch.merge_into(main_branch)
        except lakefs.exceptions.LakeFSException as e:
            logging.error(f"Failed to merge branch: {e}")

def delete_branch():
        logging.info("Deleting branch.")
        command = [
            "lakectl", "branch", "delete",
            f"lakefs://example-repo/ingestion",
            "--yes"
        ]
        if execute_lakectl_command(command, f"Failed to delete branch."):
            logging.info(f"Branch deleted.")

def main():

    initialize_lakefs_branch()
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
        bucket_path = args.bucket_path.rstrip('/') + '/'
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
        workflow_manager.run_workflows(shot_list, parallel=not args.serial)
        logging.info(f"Finished source {source}")

    # Upload and commit each shot to lakeFS
    for shot in shot_list:
        upload_shot_to_lakefs(shot, args.dataset_path, args.file_format)
        commit_shot_to_lakefs(shot)
        remove_shot_from_local(shot, args.dataset_path, args.file_format)
    merge_branch()
    delete_branch()

if __name__ == "__main__":
    main()
