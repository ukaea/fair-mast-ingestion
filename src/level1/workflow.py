import traceback
from pathlib import Path
from typing import Optional

from src.core.config import IngestionConfig
from src.core.load import loader_registry
from src.core.log import logger
from src.core.upload import UploadS3
from src.core.writer import MultiWriter, dataset_writer_registry
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
        use_uda_group_names: bool = False,
    ):
        self.config = config
        self.facility = facility
        self.include_sources = include_sources
        self.exclude_sources = exclude_sources
        self.verbose = verbose
        self.use_uda_group_names = use_uda_group_names

    def __call__(self, shot: int):
        if self.verbose:
            logger.setLevel("DEBUG")

        self.writers = [
            dataset_writer_registry.create(w.type, **w.options)
            for w in self.config.writers
        ]
        self.writer = (
            MultiWriter(self.writers) if len(self.writers) > 1 else self.writers[0]
        )
        self.loader = loader_registry.create("uda", include_error=True)
        self.pipelines = pipelines_registry.create(self.facility)

        try:
            self.create_dataset(shot)
            self.writer.finalize(f"{shot}.{self.writer.file_extension}")
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
            use_uda_group_names=self.use_uda_group_names,
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
