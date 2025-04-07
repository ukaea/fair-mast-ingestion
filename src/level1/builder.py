from typing import Optional

import xarray as xr

from src.core.load import BaseLoader, MissingMetadataError, MissingProfileError
from src.core.log import logger
from src.core.utils import harmonise_name, read_json_file
from src.core.writer import DatasetWriter
from src.level1.pipelines import Pipelines

SEG_FAULT_LIST = [
    "EPQ/INPUT/CONSTRAINTS/MSE/STRDIM_SHORTNAME",
    "EPQ/INPUT/CONSTRAINTS/MSE/SHORTNAME",
]


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
        try:
            dataset_infos = self.list_datasets(shot)
        except MissingMetadataError as e:
            logger.warning(f"Skipping shot {shot} as metadata is missing: {e}")
            return

        for dataset_info in dataset_infos:
            group_name = dataset_info.name

            logger.info(f"Loading dataset {group_name} for shot #{shot}")
            datasets = self.load_datasets(shot, group_name)

            if len(datasets) == 0:
                logger.warning(f"No datasets found for {group_name} in shot #{shot}")
                continue

            logger.info(f"Processing {group_name} for shot #{shot}")
            pipeline = self.pipelines.get(group_name)

            dataset: xr.Dataset = pipeline(datasets)
            dataset, group_name = self._rename_group(dataset, group_name)

            dataset.attrs["name"] = group_name
            dataset.attrs["description"] = dataset_info.description
            dataset.attrs["quality"] = dataset_info.quality
            dataset.attrs["license"] = {"name": "Creative Commons 4.0 BY-SA", "url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en"}

            logger.info(f"Writing {group_name} for shot #{shot}")
            file_name = f"{shot}.{self.writer.file_extension}"
            self.writer.write(file_name, group_name, dataset)

    def load_datasets(self, shot, group_name: str) -> dict[str, xr.Dataset]:
        signal_infos = self.loader.list_signals(shot)
        signal_infos = [info for info in signal_infos if info.dataset == group_name]

        if group_name == "xdc":
            xdc_accepted_signals = [
                "/XDC/IP/T/IPREF",
                "XDC_IP_T_IPREF",
                "/XDC/DENSITY/T/NELREF",
                "XDC_DENSITY_T_NELREF",
                "/XDC/GAS/F/BC11",
                "/XDC/GAS/F/BC5",
                "/XDC/GAS/F/ECEL",
                "/XDC/GAS/F/HECC",
                "/XDC/GAS/F/HFS",
                "/XDC/GAS/F/HL1",
                "/XDC/GAS/F/HL11",
                "/XDC/GAS/F/HM12A",
                "/XDC/GAS/F/HM12B",
                "/XDC/GAS/F/HU11",
                "/XDC/GAS/F/HU6",
                "/XDC/GAS/F/HU8",
                "/XDC/GAS/F/IBFLA",
                "/XDC/GAS/F/IBFLB",
                "/XDC/GAS/F/IBFUA",
                "/XDC/GAS/F/IBFUB",
                "/XDC/GAS/F/IBIL",
                "/XDC/GAS/F/TC11",
                "/XDC/GAS/F/TC5A",
                "/XDC/GAS/F/TC5B",
                "XDC_GAS_F_BC11",
                "XDC_GAS_F_BC5",
                "XDC_GAS_F_ECEL",
                "XDC_GAS_F_HECC",
                "XDC_GAS_F_HFS",
                "XDC_GAS_F_HL1",
                "XDC_GAS_F_HL11",
                "XDC_GAS_F_HM12A",
                "XDC_GAS_F_HM12B",
                "XDC_GAS_F_HU11",
                "XDC_GAS_F_HU6",
                "XDC_GAS_F_HU8",
                "XDC_GAS_F_IBFLA",
                "XDC_GAS_F_IBFLB",
                "XDC_GAS_F_IBFUA",
                "XDC_GAS_F_IBFUB",
                "XDC_GAS_F_IBIL",
                "XDC_GAS_F_TC11",
                "XDC_GAS_F_TC5A",
                "XDC_GAS_F_TC5B",
                "XDC_GAS_F_G1",
                "XDC_GAS_F_G2",
                "XDC_GAS_F_G3",
                "XDC_GAS_F_G4",
                "XDC_GAS_F_G5",
                "XDC_GAS_F_G6",
                "XDC_GAS_F_G7",
                "XDC_GAS_F_G8",
                "XDC_GAS_F_G9",
                "XDC_GAS_F_G10",
                "XDC_GAS_F_G11",
                "XDC_GAS_F_G12",
                "/XDC/SHAPE/S/S1_CNTRL",
                "/XDC/SHAPE/S/S2_CNTRL",
                "/XDC/SHAPE/S/S3_CNTRL",
                "/XDC/SHAPE/S/S4_CNTRL",
                "/XDC/SHAPE/S/S5_CNTRL",
                "/XDC/SHAPE/S/S6_CNTRL",
                "/XDC/SHAPE/S/S7_CNTRL",
                "/XDC/SHAPE/S/S8_CNTRL",
                "/XDC/Z/S/ZIP",
                "XDC_Z_S_ZIP",
                "/XDC/Z/S/ZIPREF",
                "XDC_Z_S_ZIPREF",
                "XDC/PLASMA/T/IP_REF",
                "XDC/FUELLING/T/DENSITY_REF_TARGET",
            ]

            signal_infos = [
                info for info in signal_infos if info.name in xdc_accepted_signals
            ]

        datasets = {}
        for signal_info in signal_infos:
            uda_name = signal_info.name

            if uda_name in SEG_FAULT_LIST:
                logger.warning(
                    f"Skipping {uda_name} as it is in the seg fault exclude list."
                )
                continue

            try:
                name = harmonise_name(uda_name)
                logger.debug(f"Loading {name} ({uda_name})")
                dataset = self.loader.load(shot, uda_name)
                dataset.attrs["name"] = name
                dataset.attrs["source"] = group_name
                dataset.attrs["quality"] = signal_info.quality
                dataset.attrs["license"] = {"name": "Creative Commons 4.0 BY-SA", "url": "https://creativecommons.org/licenses/by-sa/4.0/deed.en"}
                datasets[name] = dataset
            except MissingProfileError as e:
                if "StructuredData" not in str(e):
                    logger.warning(e)
        return datasets

    def _rename_group(self, dataset: xr.Dataset, group_name: str):
        if group_name in self.group_name_mapping:
            mapping = self.group_name_mapping[group_name]
            if "imas" in mapping:
                imas_name = mapping["imas"]
                dataset.attrs["imas"] = imas_name
            dataset.attrs["uda_name"] = group_name
            group_name = mapping["name"]

        return dataset, group_name

    def list_datasets(self, shot: int):
        infos = self.loader.list_datasets(shot)
        include_all = len(self.include_datasets) == 0
        infos = [
            info for info in infos if include_all or info.name in self.include_datasets
        ]
        infos = [info for info in infos if info.name not in self.exclude_datasets]
        return infos
