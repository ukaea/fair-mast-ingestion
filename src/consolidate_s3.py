import argparse
from src.uploader import UploadConfig
import subprocess
import shutil
import zarr
import logging
from dask.distributed import Client, as_completed
from dask_mpi import initialize

def consolidate(shot):
    """Consolidate the metadata for the given Zarr shot."""
    zarr.consolidate_metadata(shot)
    with zarr.open(shot) as f:
        for source in f.keys():
            zarr.consolidate_metadata(f"{shot}/{source}")
            for signal in f[source].keys():
                zarr.consolidate_metadata(f"{shot}/{source}/{signal}")

def download_shot(shot, bucket_path, local_path, config):
    """Download the Zarr file for the given shot number."""
    download_command = [
        "s5cmd",
        "--credentials-file", config.credentials_file,
        "--endpoint-url", config.endpoint_url,
        "cp", f"{bucket_path}{shot}*", local_path
    ]

    return subprocess.run(download_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def upload_shot(shot, bucket_path, local_path, config):
    """Upload the consolidated Zarr file back to S3."""
    upload_command = [
        "s5cmd",
        "--credentials-file", config.credentials_file,
        "--endpoint-url", config.endpoint_url,
        "cp", "--acl", "public-read", f"{local_path}/{shot}.zarr", bucket_path
    ]

    return subprocess.run(upload_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def process_shots(shot, local_path):
    """Process the Zarr files for the given shot number."""
    logging.info(f"Processing shot {shot}...")
    
    download_result = download_shot(shot)
    if download_result.returncode == 0:
        logging.info(f"Successfully downloaded shot {shot}")
        consolidate(f"{local_path}/{shot}.zarr")
        upload_result = upload_shot(shot)

        # Check if the upload succeeded
        if upload_result.returncode == 0:
            logging.info(f"Successfully uploaded consolidated file: {shot}.zarr")
        else:
            logging.error(f"Failed to upload {shot}.zarr: {upload_result.stderr.strip()}")

        # Remove the downloaded Zarr directory
        shutil.rmtree(f"{local_path}/{shot}.zarr")
        logging.info(f"Deleted local file: {shot}.zarr")
    else:
        logging.error(f"Failed to download {shot}: {download_result.stderr.strip()}")

if __name__ == "__main__":
    initialize()

    logging.basicConfig(
        filename="process_shots.log",  # Log to a file named 'process_shots.log'
        level=logging.INFO,            # Log level: INFO and above (INFO, WARNING, ERROR)
        format="%(asctime)s - %(levelname)s - %(message)s",  # Log format
        datefmt="%Y-%m-%d %H:%M:%S"   # Time format
    )

    parser = argparse.ArgumentParser(
        prog="Consolidate s3",
        description="Processing Zarr files",
    )

    parser.add_argument("bucket_path")
    parser.add_argument("local_path")
    parser.add_argument("--credentials_file", default=".s5cfg.stfc")
    parser.add_argument("--endpoint_url", default="https://s3.echo.stfc.ac.uk")
    parser.add_argument("--start_shot", type=int, default=11695)
    parser.add_argument("--end_shot", type=int, default=30472)

    args = parser.parse_args()

    config = UploadConfig(
            credentials_file=args.credentials_file,
            endpoint_url=args.endpoint_url,
            url=args.bucket_path,
        )

    dask_client = Client()

    shot_list = list(range(args.start_shot, args.end_shot))
    tasks = []

    # Submit tasks to the Dask cluster
    for shot in shot_list:
        task = dask_client.submit(process_shots, shot)
        tasks.append(task)

    n = len(tasks)
    for i, task in enumerate(as_completed(tasks)):
        logging.info(f"Done shot {i+1}/{n} = {(i+1)/n*100:.2f}%")
