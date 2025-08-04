import traceback
from pathlib import Path
from typing import Optional

from src.core.config import IngestionConfig
from src.core.icechunk import IcechunkUploader
from src.core.load import loader_registry
from src.core.log import logger
from src.core.upload import UploadS3
from src.core.writer import dataset_writer_registry
from src.level1.builder import DatasetBuilder
from src.level1.pipelines import pipelines_registry


class IngestionWorkflow:
    def __init__(
        self,
        config: IngestionConfig,
        facility: str,
        include_sources: Optional[list[str]] = [],
        exclude_sources: Optional[list[str]] = [],
        verbose: bool = False,
    ):
        self.config = config
        self.facility = facility
        self.include_sources = include_sources
        self.exclude_sources = exclude_sources
        self.verbose = verbose

    def __call__(self, shot: int):
        if self.verbose:
            logger.setLevel("DEBUG")

        if self.config.writer.options["zarr_format"] != 3 and self.config.icechunk:
            logger.warning(
                "Icechunk is only supported for Zarr format version 3. "
                "Please set 'zarr_format' to 3 in the config file."
            )
            return
        
        if self.config.upload and self.config.icechunk:
            logger.warning(
                "Unable to upload to S3 and version with Icechunk at the same time. Select one or the other in the config file."
            )
            return

        self.writer = dataset_writer_registry.create(
            self.config.writer.type, **self.config.writer.options
        )
        self.loader = loader_registry.create("uda", include_error=True)
        self.pipelines = pipelines_registry.create(self.facility)

        try:
            self.create_dataset(shot)
            self.icechunk_dataset(shot)
            self.upload_dataset(shot)
            logger.info(f"Done shot #{shot}")
        except Exception as e:
            logger.error(
                f"Failed to run workflow for {shot} with error {type(e)}: {e}\n"
            )
            logger.debug(traceback.print_exception(e))

    def create_dataset(self, shot: int):
        builder = DatasetBuilder(
            self.loader,
            self.writer,
            self.pipelines,
            self.include_sources,
            self.exclude_sources,
        )

        builder.create(shot)

    def upload_dataset(self, shot: int):
        if self.config.upload is None:
            return

        file_name = f"{shot}.{self.writer.file_extension}"
        local_file = self.config.writer.options["output_path"] / Path(file_name)
        remote_file = f"{self.config.upload.base_path}/"

        if not local_file.exists():
            logger.error(f"File {local_file} does not exist")
            return

        uploader = UploadS3(self.config.upload)
        uploader.upload(local_file, remote_file)

    def icechunk_dataset(self, shot: int):
        if self.config.icechunk is None:
            return
        
        file_name = f"{shot}.{self.writer.file_extension}"
        local_file = self.config.writer.options["output_path"] / Path(file_name)
        
        if not local_file.exists():
            logger.warning(f"File {local_file} does not exist")
            return
    
        icechunk = IcechunkUploader(self.config.icechunk)

        if self.config.icechunk.s3 is not None:
            logger.info("Uploading to Icechunk remote store...")
            icechunk.remote_upload(local_file, shot)
        else:
            logger.info("Uploading to Icechunk local store...")
            icechunk.local_upload(local_file, shot)
            

