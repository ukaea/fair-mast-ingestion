import os
import traceback
import shutil
import subprocess
import logging
from pathlib import Path
import xarray as xr
import pandas as pd

from src.transforms import MASTPipelineRegistry, MASTUPipelineRegistry
from src.mast import MASTClient
from src.reader import DatasetReader, SignalMetadataReader, SourceMetadataReader
from src.writer import DatasetWriter
from src.uploader import UploadConfig

logging.basicConfig(level=logging.INFO)


class CleanupDatasetTask:

    def __init__(self, path: str) -> None:
        self.path = path

    def __call__(self):
        if Path(self.path).exists():
            shutil.rmtree(self.path)
        else:
            logging.warning(f"Cannot remove path: {self.path}")


class UploadDatasetTask:

    def __init__(self, local_file: Path, config: UploadConfig):
        self.config = config
        self.local_file = local_file

    def __call__(self):
        local_file_name = str(self.local_file) + '/'
        upload_file_name = self.config.url + f"{self.local_file.parent.name}/{self.local_file.name}/"

        if not Path(local_file_name).exists():
            return
            
        logging.info(f"Uploading {self.local_file} to {upload_file_name}")

        if not Path(self.config.credentials_file).exists():
            raise RuntimeError(f"Credentials file {self.config.credentials_file} does not exist!")

        env = os.environ.copy()

        args = [
            "s5cmd",
            "--credentials-file",
            self.config.credentials_file,
            "--endpoint-url",
            self.config.endpoint_url,
            "sync",
            "--delete",
            "--acl",
            "public-read",
            local_file_name,
            upload_file_name,
        ]

        logging.debug(' ' .join(args))

        subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.STDOUT,
            env=env,
            check=True,
        )


class CreateDatasetTask:

    def __init__(
        self,
        metadata_dir: str,
        dataset_dir: str,
        shot: int,
        signal_names: list[str] = [],
        source_names: list[str] = [],
        file_format: str = 'zarr',
        facility: str = 'MAST'
    ):
        self.shot = shot
        self.metadata_dir = Path(metadata_dir)
        self.reader = DatasetReader(shot)
        self.writer = DatasetWriter(shot, dataset_dir, file_format)
        self.signal_names = signal_names
        self.source_names = source_names

        if facility == "MAST":
            self.pipelines = MASTPipelineRegistry()
        else:
            self.pipelines = MASTUPipelineRegistry()

    def __call__(self):
        try:
            self._main()
        except Exception as e:
            trace = traceback.format_exc()
            logging.error(f"Error reading sources for shot {self.shot}: {e}\n{trace}")
            
    def _main(self):
        signal_infos, source_infos = self._read_metadata()

        if source_infos is None or signal_infos is None:
            return 

        signal_infos = self._filter_signals(signal_infos)

        self.writer.write_metadata()

        for source_name, source_group_index in signal_infos.groupby("source").groups.items():
            source_info = self._get_source_metadata(source_name, source_infos)
            signal_infos_for_source = self._get_signals_for_source(source_name, source_group_index, signal_infos)
            self._process_source(source_name, signal_infos_for_source, source_info)

        self.writer.consolidate_dataset()

    def _process_source(self, source_name: str, signal_infos: pd.DataFrame, source_info: dict):
        signal_datasets = self.load_source(signal_infos)
        pipeline = self.pipelines.get(source_name)
        dataset = pipeline(signal_datasets)
        dataset.attrs.update(source_info)
        self.writer.write_dataset(dataset)

    def _get_source_metadata(self, source_name, source_infos: pd.DataFrame) -> dict:
        source_info = source_infos.loc[source_infos["name"] == source_name].iloc[0]
        source_info = source_info.to_dict()
        return source_info

    def _get_signals_for_source(self, source_name: str, source_group_index: pd.Series, signal_infos: pd.DataFrame):
        signal_infos_for_source = signal_infos.loc[source_group_index]
        if source_name == 'xdc':
            # Drop any CPU which is not CPU1 or isoflux
            # names = ['xdc1', 'xdc2', 'xdc3', 'xdc4', 'cpu2', 'cpu3', 'cpu4', 'isoflux']
            names = ['ip_t_ipref', 'density_t_nelref', 'ai_raw_tf_current', 'shape']
            name_mask = signal_infos_for_source['name'].map(lambda x: any([c in x for c in names]))
            signal_infos_for_source = signal_infos_for_source.loc[name_mask]
        return signal_infos_for_source

    def _read_metadata(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        try:
            signal_infos = self.read_signal_info()
            source_infos = self.read_source_info()
        except FileNotFoundError:
            message = f"Could not find source/signal metadata file for shot {self.shot}"
            logging.warning(message)
            return None, None
        
        return signal_infos, source_infos


    def _filter_signals(self, signal_infos: pd.DataFrame) -> pd.DataFrame:
        if len(self.signal_names) > 0:
            signal_infos = signal_infos.loc[signal_infos.name.isin(self.signal_names)]

        if len(self.source_names) > 0:
            signal_infos = signal_infos.loc[signal_infos.source.isin(self.source_names)]

        return signal_infos

    def load_source(self, group: pd.DataFrame) -> dict[str, xr.Dataset]:
        datasets = {}

        for _, info in group.iterrows():
            info = info.to_dict()
            name = info["name"]
            format = info["format"]
            format = format if format is not None else ""

            try:
                client = MASTClient()
                if info["signal_type"] != "Image":
                    dataset = client.get_signal(
                        shot_num=self.shot, name=info["uda_name"], format=format
                    )
                else:
                    dataset = client.get_image(
                        shot_num=self.shot, name=info["uda_name"]
                    )
            except Exception as e:
                uda_name = info["uda_name"]
                logging.warning(f"Could not read dataset {name} ({uda_name}) for shot {self.shot}: {e}")
                continue

            dataset.attrs.update(info)
            dataset.attrs["dims"] = list(dataset.sizes.keys())
            datasets[name] = dataset
        return datasets

    def read_signal_info(self) -> pd.DataFrame:
        return pd.read_parquet(self.metadata_dir / f"signals/{self.shot}.parquet")

    def read_source_info(self) -> pd.DataFrame:
        return pd.read_parquet(self.metadata_dir / f"sources/{self.shot}.parquet")


class CreateSignalMetadataTask:
    def __init__(self, data_dir: str, shot: int):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.shot = shot
        self.reader = SignalMetadataReader(shot)

    def __call__(self):
        df = self.reader.read_metadata()
        if len(df) > 0:
            df.to_parquet(self.data_dir / f"{self.shot}.parquet")


class CreateSourceMetadataTask:
    def __init__(self, data_dir: str, shot: int):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True, parents=True)
        self.shot = shot
        self.reader = SourceMetadataReader(shot)

    def __call__(self):
        df = self.reader.read_metadata()
        if len(df) > 0:
            df.to_parquet(self.data_dir / f"{self.shot}.parquet")
