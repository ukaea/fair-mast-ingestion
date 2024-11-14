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

def download_shot(shot):
    """Download the Zarr file for the given shot number."""
    download_command = [
        "s5cmd",
        "--credentials-file", ".s5cfg.stfc",
        "--endpoint-url", "https://s3.echo.stfc.ac.uk",
        "cp", f"s3://mast/level1/shots/{shot}*", "/rds/project/rds-mOlK9qn0PlQ/fairmast"
    ]

    return subprocess.run(download_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def upload_shot(shot):
    """Upload the consolidated Zarr file back to S3."""
    upload_command = [
        "s5cmd",
        "--credentials-file", ".s5cfg.stfc",
        "--endpoint-url", "https://s3.echo.stfc.ac.uk",
        "cp", "--acl", "public-read", f"/rds/project/rds-mOlK9qn0PlQ/fairmast/{shot}.zarr", "s3://mast/level1/shots/"
    ]

    return subprocess.run(upload_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

def process_shots(shot):
    """Process the Zarr files for the given shot number."""
    logging.info(f"Processing shot {shot}...")
    
    download_result = download_shot(shot)
    if download_result.returncode == 0:
        logging.info(f"Successfully downloaded shot {shot}")
        consolidate(f"/rds/project/rds-mOlK9qn0PlQ/fairmast/{shot}.zarr")
        upload_result = upload_shot(shot)

        # Check if the upload succeeded
        if upload_result.returncode == 0:
            logging.info(f"Successfully uploaded consolidated file: {shot}.zarr")
        else:
            logging.error(f"Failed to upload {shot}.zarr: {upload_result.stderr.strip()}")

        # Remove the downloaded Zarr directory
        shutil.rmtree(f"/rds/project/rds-mOlK9qn0PlQ/fairmast/{shot}.zarr")
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

    dask_client = Client()

    shot_list = list(range(11695, 30472))
    tasks = []

    # Submit tasks to the Dask cluster
    for shot in shot_list:
        task = dask_client.submit(process_shots, shot)
        tasks.append(task)

    n = len(tasks)
    for i, task in enumerate(as_completed(tasks)):
        logging.info(f"Done shot {i+1}/{n} = {(i+1)/n*100:.2f}%")
