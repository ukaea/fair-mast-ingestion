from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel


class UploadConfig(BaseModel):
    base_path: str
    credentials_file: str
    endpoint_url: str


class WriterConfig(BaseModel):
    type: str
    options: dict[str, Any]


class ReaderConfig(BaseModel):
    type: str
    options: Optional[dict[str, Any]] = {}

class IcechunkConfig(BaseModel):
    icechunk_repo_path: str
    icechunk_branch: str = "main"
    commit_message: str

class IngestionConfig(BaseModel):
    upload: Optional[UploadConfig] = None
    icechunk: Optional[IcechunkConfig] = None
    readers: dict[str, ReaderConfig]
    writer: WriterConfig


def load_config(config_file: str) -> IngestionConfig:
    with Path(config_file).open("r") as handle:
        config = yaml.load(handle, yaml.FullLoader)
        config = IngestionConfig.model_validate(config)
    return config
