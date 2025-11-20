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
        
    def remote_upload_from_memory(self, data_tree: xr.DataTree, shot: int):
        """Upload data directly from memory to remote icechunk store without writing to disk first."""
        
        logger.info(f"Writing Zarr data from memory to s3://{self.config.s3.bucket}/{self.config.s3.prefix}{shot}")
        
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
        
        data_tree.to_zarr(session.store, mode="a", consolidated=False, compute=False)

        if self.config.commit_message is None:
            self.config.commit_message = f"Upload shot {shot} to S3 Icechunk repo"
        
        snapshot = session.commit(self.config.commit_message)
        logger.info(f"Icechunk commit completed. Snapshot: {snapshot}")