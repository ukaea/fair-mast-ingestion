from pathlib import Path
from typing import Union
from importlib.resources import files

from src.core.load import BaseLoader, loader_registry
from src.level2.reader import DatasetReader
from src.core.model import Mapping, load_model


def load_mapping(mapping_file: str) -> Mapping:
    mapping = load_model(mapping_file)
    return mapping


def create_loader(name: str) -> BaseLoader:
    loader = loader_registry.create(name)
    return loader


def create_reader(
    mapping: Union[str, Mapping], loader: Union[str, BaseLoader]
) -> DatasetReader:
    if isinstance(mapping, str):
        if Path(mapping).exists():
            mapping = load_mapping(mapping)
        else:
            file_path = files("src.mappings")
            mapping = load_mapping(file_path / f"{mapping}.yml")

    if isinstance(loader, str):
        loader = create_loader(loader)

    reader = DatasetReader(mapping, loader)
    return reader
