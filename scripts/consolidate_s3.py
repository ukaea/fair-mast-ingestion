import argparse
import logging
import shutil
import subprocess

import zarr
from pathlib import Path
from src.core.workflow_manager import WorkflowManager
from src.core.config import UploadConfig


def consolidate(shot):
    """Consolidate the metadata for the given Zarr shot."""
    zarr.consolidate_metadata(shot)
    with zarr.open(shot) as f:
        for source in f.keys():
            zarr.consolidate_metadata(f"{shot}/{source}")


def download_shot(shot, local_path, config):
    """Download the Zarr file for the given shot number."""
    download_command = [
        "s5cmd",
        "--credentials-file",
        config.credentials_file,
        "--endpoint-url",
        config.endpoint_url,
        "cp",
        f"{config.base_path}{shot}*",
        local_path,
    ]

    return subprocess.run(
        download_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def upload_shot(shot, local_path, config):
    """Upload the consolidated Zarr file back to S3."""
    upload_command = [
        "s5cmd",
        "--credentials-file",
        config.credentials_file,
        "--endpoint-url",
        config.endpoint_url,
        "cp",
        "--acl",
        "public-read",
        f"{local_path}/{shot}.zarr",
        config.base_path,
    ]

    return subprocess.run(
        upload_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )


def process_shots(shot, local_path, config):
    """Process the Zarr files for the given shot number."""
    logging.info(f"Processing shot {shot}...")

    download_result = download_shot(shot, local_path, config)
    if download_result.returncode == 0:
        logging.info(f"Successfully downloaded shot {shot}")
        consolidate(f"{local_path}/{shot}.zarr")
        upload_result = upload_shot(shot, local_path, config)

        # Check if the upload succeeded
        if upload_result.returncode == 0:
            logging.info(f"Successfully uploaded consolidated file: {shot}.zarr")
        else:
            logging.error(
                f"Failed to upload {shot}.zarr: {upload_result.stderr.strip()}"
            )

        # Remove the downloaded Zarr directory
        shutil.rmtree(f"{local_path}/{shot}.zarr")
        logging.info(f"Deleted local file: {shot}.zarr")
    else:
        logging.error(f"Failed to download {shot}: {download_result.stderr.strip()}")


if __name__ == "__main__":
    logging.basicConfig(
        filename="process_shots.log",  # Log to a file named 'process_shots.log'
        level=logging.INFO,  # Log level: INFO and above (INFO, WARNING, ERROR)
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S",  # Time format
    )

    parser = argparse.ArgumentParser(
        prog="Consolidate s3",
        description="Processing Zarr files",
    )

    parser.add_argument("bucket_path")
    parser.add_argument("local_path")
    parser.add_argument("-n", "--n-workers", type=int, default=None)
    parser.add_argument("--credentials-file", default=".s5cfg.stfc")
    parser.add_argument("--endpoint_url", default="https://s3.echo.stfc.ac.uk")
    parser.add_argument("--start-shot", type=int, default=11695)
    parser.add_argument("--end-shot", type=int, default=30472)

    args = parser.parse_args()

    config = UploadConfig(
        credentials_file=args.credentials_file,
        endpoint_url=args.endpoint_url,
        base_path=args.bucket_path,
    )

    Path(args.local_path).mkdir(exist_ok=True, parents=True)

    shot_list = list(range(args.start_shot, args.end_shot))
    tasks = []

    workflow_manager = WorkflowManager(process_shots)
    workflow_manager.run_workflows(
        shot_list, args.n_workers, local_path=args.local_path, config=config
    )
