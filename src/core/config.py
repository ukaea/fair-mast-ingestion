from pathlib import Path
from typing import Any, Optional

import yaml
from pydantic import BaseModel, model_validator


class UploadConfig(BaseModel):
    base_path: str
    credentials_file: str
    endpoint_url: str


class WriterConfig(BaseModel):
    type: str
    options: dict[str, Any] = {}

    @property
    def is_s3(self) -> bool:
        """A writer publishes directly to S3 when its output path is an s3:// URI."""
        output_path = str(self.options.get("output_path", ""))
        return output_path.startswith("s3://")


class ReaderConfig(BaseModel):
    type: str
    options: Optional[dict[str, Any]] = {}


class IngestionConfig(BaseModel):
    upload: Optional[UploadConfig] = None
    readers: dict[str, ReaderConfig]
    writers: list[WriterConfig]

    @model_validator(mode="before")
    @classmethod
    def _coerce_singular_writer(cls, data: Any) -> Any:
        """Accept the legacy singular ``writer:`` key as a one-element ``writers`` list."""
        if isinstance(data, dict) and "writer" in data and "writers" not in data:
            data = dict(data)
            data["writers"] = [data.pop("writer")]
        return data


def load_config(config_file: str) -> IngestionConfig:
    with Path(config_file).open("r") as handle:
        config = yaml.load(handle, yaml.FullLoader)
        config = IngestionConfig.model_validate(config)
    return config
