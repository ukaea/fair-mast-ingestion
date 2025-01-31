import argparse
import logging
from pathlib import Path

import s3fs
import xarray as xr
import zarr

from src.core.log import logger
from src.core.metadata import ParquetMetadataWriter
from src.core.workflow_manager import WorkflowManager


class ShotMetadataParser:
    def __init__(self, output_path: str, bucket_path: str, endpoint_url: str):
        self.bucket_path = bucket_path
        self.endpoint_url = endpoint_url
        self.fs = s3fs.S3FileSystem(anon=True, endpoint_url=endpoint_url)

        self.output_path = Path(output_path)
        self.output_path.mkdir(exist_ok=True, parents=True)
        (self.output_path / "signals").mkdir(exist_ok=True)
        (self.output_path / "sources").mkdir(exist_ok=True)

    def __call__(self, shot: int):
        path = f"{self.bucket_path}/{shot}.zarr"
        store = zarr.storage.FSStore(path, fs=self.fs)
        writer = ParquetMetadataWriter(self.output_path, self.bucket_path)

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

        writer.save(shot)
        logger.info(f"Done shot {shot}!")


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
    parser.add_argument("--output-path", type=str, default="./metadata")
    parser.add_argument("-n", "--n-workers", type=int, default=4)

    args = parser.parse_args()

    shots = range(args.shot_min, args.shot_max + 1)

    db_path = args.output_path
    db_path = Path(db_path).absolute()

    metadata_parser = ShotMetadataParser(db_path, args.bucket_path, args.endpoint_url)
    workflow_manager = WorkflowManager(metadata_parser)
    workflow_manager.run_workflows(shots, n_workers=args.n_workers)


if __name__ == "__main__":
    main()
