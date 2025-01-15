import traceback
from pathlib import Path
from typing import Optional

from src.core.config import IngestionConfig
from src.core.load import loader_registry
from src.core.log import logger
from src.core.metadata import MetadataWriter
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

        writer_config = self.config.writer
        self.writer = dataset_writer_registry.create(
            writer_config.type, **writer_config.options
        )
        self.loader = loader_registry.create("uda", include_error=True)
        self.pipelines = pipelines_registry.create(self.facility)

        db_path = Path(self.config.metadatabase_file).absolute()
        uri = f"sqlite:////{db_path}"
        if self.config.upload is not None:
            remote_path = f"{self.config.upload.base_path}/"
        else:
            remote_path = self.writer.output_path
        self.metadata_writer = MetadataWriter(uri, remote_path)

        try:
            self.create_dataset(shot)
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
            self.metadata_writer,
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
            logger.warning(f"File {local_file} does not exist")
            return

        uploader = UploadS3(self.config.upload)
        uploader.upload(local_file, remote_file)
