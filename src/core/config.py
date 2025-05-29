from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, validator


class UploadConfig(BaseModel):
    base_path: str
    credentials_file: str
    endpoint_url: str

class IcechunkUploadConfig(BaseModel):
    bucket: str
    prefix: str
    endpoint_url: str
    access_key_id: str
    secret_access_key: str

    @validator("prefix", pre=True)
    def normalize_prefix(cls, v):
        if v.startswith("/"):
            v = v[1:]
        if not v.endswith("/"):
            v += "/"
        return v

class WriterConfig(BaseModel):
    type: str
    options: dict[str, Any]


class ReaderConfig(BaseModel):
    type: str
    options: Optional[dict[str, Any]] = {}

class IcechunkConfig(BaseModel):
    upload: Optional[IcechunkUploadConfig] = None
    local_icechunk_repo_path: Optional[str] = None
    icechunk_branch: str = "main"
    commit_message: str = None

class IngestionConfig(BaseModel):
    icechunk: Optional[IcechunkConfig] = None
    readers: dict[str, ReaderConfig]
    writer: WriterConfig


def load_config(config_file: str) -> IngestionConfig:
    with Path(config_file).open("r") as handle:
        config = yaml.load(handle, yaml.FullLoader)
        config = IngestionConfig.model_validate(config)
    return config
