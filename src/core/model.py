import json
from collections import OrderedDict
from enum import Enum
from pathlib import Path
from typing import Any, Optional, Union

import yaml
from pydantic import BaseModel, Field, field_validator


class FillOptions(str, Enum):
    FFILL = "ffill"
    BFILL = "bfill"
    NONE = "none"


class InterpolationParams(BaseModel):
    start: Optional[float] = None
    end: Optional[float] = None
    step: Optional[float] = None
    method: Optional[str] = "zero"
    fill: Optional[FillOptions] = None
    dropna: Optional[bool] = None


class ShotRange(BaseModel):
    shot_min: int = Field(gt=0)
    shot_max: int = Field(gt=0)


class BackgroundCorrection(BaseModel):
    tmin: int
    tmax: int


class Channel(BaseModel):
    name: str
    scale: float = 1.0


class Source(BaseModel):
    name: str
    shot_range: Optional[ShotRange] = None
    channels: Optional[list[Channel]] = None
    background_correction: Optional[BackgroundCorrection] = None
    attributes: Optional[dict] = None

    @field_validator("channels", mode="before")
    @classmethod
    def _coerce_channels(cls, v):
        # allow channels to be dicts to specify scale factors
        return [_coerce_channel(e) for e in v] if v is not None else v

SourceType = Union[list[Source], str]


class Dimension(BaseModel):
    source: Optional[SourceType] = None
    imas: Optional[str] = None
    units: Optional[str] = None
    scale: float = 1.0
    target_units: Optional[str] = None

class Geometry(BaseModel):
    stem: Optional[str] = None
    path: Optional[str] = None
    shot: Optional[str] = None
    measurement: Optional[str] = None
    channel_name: Optional[str] = "geometry_channel"


class ProfileInfo(BaseModel):
    source: SourceType
    imas: Optional[str] = None
    dimensions: OrderedDict[str, Optional[Dimension]]
    geometry: Optional[Geometry] = None
    units: Optional[str] = None
    scale: float = 1.0
    fill_value: Optional[float] = None
    target_units: Optional[str] = None
    description: Optional[str] = ""
    dimension_order: Optional[list[str]] = None

    class Config:
        arbitrary_types_allowed = True


class DatasetInfo(BaseModel):
    imas: Optional[str] = None
    profiles: dict[str, ProfileInfo]
    transforms: Optional[dict[str, Any]] = None
    interpolate: Optional[dict[str, InterpolationParams]] = None
    description: Optional[str] = ""

class License(BaseModel):
    name: str
    url: str

class Mapping(BaseModel):
    facility: str
    default_loader: str
    plasma_current: str
    dataset_defaults: Optional[dict[str, str]] = None
    datasets: dict[str, DatasetInfo]
    default_start: Optional[float] = None
    tmax: Optional[float] = None
    license: Optional[License] = None


def load_yaml(config_file: Union[str, Path]) -> dict[str, Any]:
    with Path(config_file).open("r") as f:
        config = yaml.load(f, yaml.FullLoader)
    return config


def load_json(file_name: Union[str, Path]):
    with Path(file_name).open("r") as f:
        data = json.load(f)
    return data


def load_model(config_file: Union[str, Path]) -> Mapping:
    config = load_yaml(config_file)
    return Mapping.model_validate(config)

def _coerce_channel(entry: Union[str, dict]) -> Channel:
    if isinstance(entry, str):
        return Channel(name=entry)
    if isinstance(entry, dict) and len(entry) == 1:
        name, config = next(iter(entry.items()))
        config = config or {}
        if not isinstance(config, dict):
            raise ValueError(f"Invalid channel entry: {entry!r}")
        return Channel(name=str(name), **config)
    raise ValueError(f"Invalid channel entry: {entry!r}")