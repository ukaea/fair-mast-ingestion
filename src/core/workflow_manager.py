from typing import Optional

from dask import config
from distributed import Client, LocalCluster, as_completed

from src.core.log import logger


class WorkflowManager:
    def __init__(self, workflow):
        self.workflow = workflow

    def run_workflows(self, shot_list: list[int], n_workers: int = 4, **kwargs):
        client = self.initialize_client(n_workers)
        tasks = []

        for shot in reversed(sorted(shot_list)):
            task = client.submit(self.workflow, shot, **kwargs)
            tasks.append(task)

        n = len(tasks)
        for i, task in enumerate(as_completed(tasks)):
            task.release()
            logger.info(f"Done shot {i+1}/{n} = {(i+1)/n*100:.2f}%")

    def initialize_client(self, n_workers: Optional[int] = None) -> Client:
        config.set({"distributed.scheduler.locks.lease-timeout": "inf"})

        try:
            # Try and get MPI, if not use dask
            from mpi4py import MPI  # type: ignore

            comm = MPI.COMM_WORLD
            size = comm.Get_size()
            rank = comm.Get_rank()
        except ImportError:
            size = None

        if size is not None and size > 1:
            # Using dask MPI client
            from dask_mpi import initialize  # type: ignore

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
