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
            self.config.credentials_file,
            "--endpoint-url",
            self.config.endpoint_url,
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
                stdout=subprocess.DEVNULL,
                stderr=subprocess.STDOUT,
                env=env,
                capture_output=True,
                check=True,
            )
        except subprocess.CalledProcessError as e:
            logger.error("Command failed with error code:", e.returncode)
            logger.error("Error message:", e.stderr)
            raise e
