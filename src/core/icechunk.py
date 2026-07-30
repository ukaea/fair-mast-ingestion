import warnings

import icechunk
import xarray as xr
from icechunk.xarray import to_icechunk

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
        repo_config = icechunk.RepositoryConfig(
            storage=icechunk.StorageSettings(
                unsafe_use_conditional_update=False,
                unsafe_use_conditional_create=False,
            )
        )

        try:
            repo = icechunk.Repository.open_or_create(storage=storage, config=repo_config)
        except icechunk.IcechunkError:
            repo = icechunk.Repository.open(storage=storage, config=repo_config)

        session = repo.writable_session(self.config.icechunk_branch)

        for group_name, node in data_tree.children.items():
            try:
                to_icechunk(node.ds, session, group=group_name)
            except FileExistsError:
                logger.info(f"Data already exists for group '{group_name}' in shot {shot}, skipping")
                continue

        if not session.has_uncommitted_changes:
            logger.info(f"No new changes for shot {shot}, skipping commit")
            return

        commit_message = self.config.commit_message or f"Upload shot {shot} to S3 Icechunk repo"

        snapshot = session.commit(commit_message)
        logger.info(f"Icechunk commit completed. Snapshot: {snapshot}")