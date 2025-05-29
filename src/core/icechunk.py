import icechunk
import icechunk.xarray
import xarray as xr
import warnings
import shutil

from src.core.log import logger
from src.core.config import IcechunkConfig

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
        data_tree.to_zarr(session.store, mode="w")

        if self.config.commit_message is None:
            snapshot = self.config.commit_message = f"Upload {local_file} to local Icechunk repo"

        snapshot = session.commit(self.config.commit_message)
        logger.info(f"Icechunk commit completed. Snapshot: {snapshot}")

        logger.info(f"Cleanup local file...")
        shutil.rmtree(local_file)


    def remote_upload(self, local_file: str, shot):
        logger.info(f"Writing Zarr data from '{local_file}' to s3://{self.config.upload.bucket}{self.config.upload.prefix}{shot}")
        
        storage = icechunk.s3_storage(
            bucket=self.config.upload.bucket,
            prefix=f"{self.config.upload.prefix}{shot}",
            endpoint_url=self.config.upload.endpoint_url,
            force_path_style=True,
            access_key_id=self.config.upload.access_key_id,
            secret_access_key=self.config.upload.secret_access_key,
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
        
        with session.allow_pickling():
            data.to_zarr(session.store, mode="w")

        if self.config.commit_message is None:
            self.config.commit_message = f"Upload {local_file} to S3 Icechunk repo"
        
        snapshot = session.commit(self.config.commit_message)
        logger.info(f"Icechunk commit completed. Snapshot: {snapshot}")

        logger.info(f"Cleanup local file...")
        shutil.rmtree(local_file)