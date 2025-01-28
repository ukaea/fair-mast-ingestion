from pathlib import Path
import xarray as xr
import argparse
import logging

import s3fs
import zarr

from src.core.log import logger
from src.core.metadata import MetadataWriter
from src.core.workflow_manager import WorkflowManager


class ShotMetadataParser:
    def __init__(self, db_path: str, bucket_path: str, endpoint_url: str):
        self.bucket_path = bucket_path
        self.endpoint_url = endpoint_url
        self.fs = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint_url)
        self.db_uri = f"sqlite:////{db_path}"

    def __call__(self, shot: int):
        path = f"{self.bucket_path}/{shot}.zarr"
        store = zarr.storage.FSStore(path, fs=self.fs)
        writer = MetadataWriter(self.db_uri, self.bucket_path)

        logger.info(f"Processing shot {shot}")

        try:
            with zarr.open_consolidated(store) as f:
                for source in f.keys():
                    logger.debug(f"Writing metadata for {source} from shot {shot}")
                    dataset = xr.open_zarr(store, group=source)
                    writer.write(shot, dataset)
        except KeyError:
            logger.info(f"Skipping {shot} as it does not exist.")
            return


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="UDA Archive Parser",
        description="Parse the MAST archive and writer to Zarr files. Upload to S3",
    )

    parser.add_argument("--shot-min", type=int, default=None)
    parser.add_argument("--shot-max", type=int, default=None)
    parser.add_argument(
        "--endpoint-url", type=str, default="https://s3.echo.stfc.ac.uk"
    )
    parser.add_argument("--bucket-path", type=str, default="s3://mast/level2/shots")
    parser.add_argument("--output-file", type=str, default="./metadata.db")
    parser.add_argument("-n", "--n-workers", type=int, default=None)

    args = parser.parse_args()

    shots = range(args.shot_min, args.shot_max)

    db_path = args.output_file
    db_path = Path(db_path).absolute()

    metadata_parser = ShotMetadataParser(db_path, args.bucket_path, args.endpoint_url)
    workflow_manager = WorkflowManager(metadata_parser)
    workflow_manager.run_workflows(shots, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
