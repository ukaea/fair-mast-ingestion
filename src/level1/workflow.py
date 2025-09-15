import time
import traceback
from pathlib import Path
from typing import Optional

from src.core.config import IngestionConfig
from src.core.icechunk import IcechunkUploader
from src.core.load import loader_registry
from src.core.log import logger
from src.core.writer import dataset_writer_registry, InMemoryDatasetWriter
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

        if self.config.icechunk is not None:
            self.writer = InMemoryDatasetWriter()
        else:
            self.writer = dataset_writer_registry.create(
                self.config.writer.type, **self.config.writer.options
            )
        self.loader = loader_registry.create("uda", include_error=True)
        self.pipelines = pipelines_registry.create(self.facility)

        try:
            self.create_dataset(shot)
            self.icechunk_dataset(shot)
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


    def icechunk_dataset(self, shot: int):
        if self.config.icechunk is None:
            return
        
        file_name = f"{shot}.{self.writer.file_extension}"
        data_tree = self.writer.get_datatree(file_name)
        
        if len(data_tree.children) == 0:
            logger.warning(f"No datasets available in memory for shot {shot}")
            return
        
        icechunk = IcechunkUploader(self.config.icechunk)
        
        if self.config.icechunk.s3 is not None:
            logger.info("Uploading to Icechunk remote store from memory...")
            icechunk.remote_upload_from_memory(data_tree, shot)

        logger.info(f"Icechunk upload completed for shot {shot}")
            
        # Clear datasets from memory after upload
        self.writer.clear_datasets(file_name)
            

