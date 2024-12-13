import traceback
from typing import Optional
from dask import config
from distributed import Client, LocalCluster, as_completed

from src.log import logger
from src.config import IngestionConfig
from src.builder import DatasetBuilder
from src.pipelines import pipelines_registry
from src.load import loader_registry
from src.writer import dataset_writer_registry


class IngestionWorkflow:
    def __init__(
        self,
        config: IngestionConfig,
        facility: str,
        include_sources: Optional[list[str]] = [],
        exclude_sources: Optional[list[str]] = [],
    ):
        self.config = config
        self.facility = facility
        self.include_sources = include_sources
        self.exclude_sources = exclude_sources

    def __call__(self, shot: int):
        try:
            self.create_dataset(shot)
            self.upload_dataset(shot)
            self.cleanup(shot)
            logger.info(f"Done shot #{shot}")
        except Exception as e:
            logger.error(f"Failed to run workflow with error {type(e)}: {e}\n")
            logger.debug(traceback.print_exception(e))

    def create_dataset(self, shot: int):
        writer_config = self.config.writer
        writer = dataset_writer_registry.create(
            writer_config.type, **writer_config.options
        )

        loader = loader_registry.create("uda")
        pipelines = pipelines_registry.create(self.facility)

        builder = DatasetBuilder(
            loader,
            writer,
            pipelines,
            self.include_sources,
            self.exclude_sources,
        )

        builder.create(shot)

    def upload_dataset(self, shot: int):
        if self.config.upload is None:
            return

    def cleanup(self, shot: int):
        pass


class WorkflowManager:
    def __init__(self, workflow):
        self.workflow = workflow

    def run_workflows(self, shot_list: list[int], n_workers: int = 4):
        client = self.initialize_client(n_workers)
        tasks = []

        for shot in shot_list:
            task = client.submit(self.workflow, shot)
            tasks.append(task)

        n = len(tasks)
        for i, task in enumerate(as_completed(tasks)):
            task.release()
            logger.info(f"Done shot {i+1}/{n} = {(i+1)/n*100:.2f}%")

    def initialize_client(self, n_workers: Optional[int] = None) -> Client:
        config.set({"distributed.scheduler.locks.lease-timeout": "inf"})

        try:
            # Try and get MPI, if not use dask
            from mpi4py import MPI

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
        except ImportError:
            size = None

        if size is not None and size > 1:
            # Using dask MPI client
            from dask_mpi import initialize

            initialize()
            client = Client()
            if rank == 0:
                logger.info(f"Running in parallel with mpi and {size} ranks")
        else:
            # Using plain dask client
            cluster = LocalCluster(n_workers=n_workers, threads_per_worker=1)
            client = Client(cluster)
            num_workers = len(client.scheduler_info()["workers"])
            logger.info(
                f"Running in parallel with dask local cluster and {num_workers} ranks"
            )

        return client
