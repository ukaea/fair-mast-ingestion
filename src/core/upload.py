import os
import shutil
import subprocess

from src.core.config import UploadConfig
from src.core.log import logger


class UploadS3:
    def __init__(self, config: UploadConfig, mode: str = "s5cmd"):
        self.config = config
        self.mode = mode

    def upload(self, local_file: str, remote_file: str):
        logger.info(f'Uploading "{local_file}" to "{remote_file}"')
        self.upload_s5cmd(local_file, remote_file)
        # clean up local file
        shutil.rmtree(local_file)

    def upload_s5cmd(self, local_file, remote_file):
        env = os.environ.copy()

        args = [
            "s5cmd",
            "--credentials-file",
            str(self.config.credentials_file),
            "--endpoint-url",
            str(self.config.endpoint_url),
            "cp",
            "--acl",
            "public-read",
            str(local_file),
            str(remote_file),
        ]

        logger.debug(" ".join(args))
        try:
            subprocess.run(
                args,
                env=env,
                capture_output=True,
                check=True,
            )
            logger.info(f"Uploaded {local_file} to {remote_file} successfully.")
        except subprocess.CalledProcessError as e:
            logger.error(f"Command failed with error code: {e.returncode}")
            logger.error(f"Error message: {e.stderr.decode('utf-8')}")
            raise e
