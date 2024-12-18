from typing import Optional

import xarray as xr

from src.load import BaseLoader, MissingProfileError
from src.log import logger
from src.pipelines import Pipelines
from src.utils import harmonise_name, read_json_file
from src.writer import DatasetWriter


class DatasetBuilder:
    def __init__(
        self,
        loader: BaseLoader,
        writer: DatasetWriter,
        pipelines: Pipelines,
        include_datasets: Optional[list[str]],
        exclude_datasets: Optional[list[str]],
    ):
        self.writer = writer
        self.pipelines = pipelines
        self.loader = loader
        self.include_datasets = include_datasets
        self.exclude_datasets = exclude_datasets
        self.group_name_mapping = read_json_file(self.pipelines.group_mapping_file)

    def create(self, shot: int):
        dataset_infos = self.list_datasets(shot)

        for dataset_info in dataset_infos:
            group_name = dataset_info.name

            logger.info(f"Loading dataset {group_name} for shot #{shot}")
            datasets = self.load_datasets(shot, group_name)

            logger.info(f"Processing {group_name} for shot #{shot}")
            pipeline = self.pipelines.get(group_name)

            dataset = pipeline(datasets)

            # rename groups
            if group_name in self.group_name_mapping:
                mapping = self.group_name_mapping[group_name]
                imas_name = mapping["imas"]

                for dataset in datasets.values():
                    dataset.attrs["imas"] = imas_name

                group_name = mapping["name"]

            logger.info(f"Writing {group_name} for shot #{shot}")
            file_name = f"{shot}.{self.writer.file_extension}"
            self.writer.write(file_name, group_name, dataset)

    def load_datasets(self, shot, group_name: str) -> dict[str, xr.Dataset]:
        signal_infos = self.loader.list_signals(shot)
        signal_infos = [info for info in signal_infos if info.dataset == group_name]
        datasets = {}
        for signal_info in signal_infos:
            name = signal_info.name
            try:
                new_name = harmonise_name(name)
                logger.debug(f"Loading {new_name}")
                dataset = self.loader.load(shot, name)
                dataset.attrs["name"] = new_name
                dataset.attrs["source"] = group_name
                datasets[new_name] = dataset
            except MissingProfileError as e:
                logger.warning(e)
        return datasets

    def list_datasets(self, shot: int):
        infos = self.loader.list_datasets(shot)
        include_all = len(self.include_datasets) == 0
        infos = [
            info for info in infos if include_all or info.name in self.include_datasets
        ]
        infos = [info for info in infos if info.name not in self.exclude_datasets]
        return infos
