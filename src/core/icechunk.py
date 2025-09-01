import shutil
import warnings

import icechunk
import icechunk.xarray
import xarray as xr

from src.core.config import IcechunkConfig
from src.core.log import logger

warnings.filterwarnings("ignore")

class IcechunkUploader:
    def __init__(self, config: IcechunkConfig,):
        self.config = config

    def local_upload(self, local_file: str, shot):
        logger.info(f"Uploading with Icechunk to repo at '{self.config.local_icechunk_repo_path}'")

        storage = icechunk.local_filesystem_storage(f"{self.config.local_icechunk_repo_path}/{shot}")
        repo = icechunk.Repository.open_or_create(storage=storage)

        session = repo.writable_session(self.config.icechunk_branch)
        logger.info(f"Writing Zarr data from '{local_file}' to Icechunk local store...")

        data_tree = xr.open_datatree(local_file)
        data_tree.to_zarr(session.store, mode="a", consolidated=False)

        if self.config.commit_message is None:
            snapshot = self.config.commit_message = f"Upload {local_file} to local Icechunk repo"

        snapshot = session.commit(self.config.commit_message)
        logger.info(f"Icechunk commit completed. Snapshot: {snapshot}")

        logger.info("Cleanup local file...")
        shutil.rmtree(local_file)


    def remote_upload(self, local_file: str, shot):
        logger.info(f"Writing Zarr data from '{local_file}' to s3://{self.config.s3.bucket}/{self.config.s3.prefix}{shot}")
        
        storage = icechunk.s3_storage(
            bucket=self.config.s3.bucket,
            prefix=f"{self.config.s3.prefix}{shot}",
            endpoint_url=self.config.s3.endpoint_url,
            force_path_style=True,
            access_key_id=self.config.s3.access_key_id,
            secret_access_key=self.config.s3.secret_access_key,
        )

        # needed for CEPH storage
        config = icechunk.RepositoryConfig(
        storage = icechunk.StorageSettings(
            unsafe_use_conditional_update=False,
            unsafe_use_conditional_create=False,
        )
        )
        
        repo = icechunk.Repository.open_or_create(storage=storage, config=config)

        session = repo.writable_session(self.config.icechunk_branch)

        data = xr.open_datatree(local_file, chunks={})
        
        fork = session.fork()
        data.to_zarr(fork.store, mode="a", consolidated=False)

        if self.config.commit_message is None:
            self.config.commit_message = f"Upload {local_file} to S3 Icechunk repo"
        
        snapshot = session.commit(self.config.commit_message)
        logger.info(f"Icechunk commit completed. Snapshot: {snapshot}")

        logger.info("Cleanup local file...")
        shutil.rmtree(local_file)