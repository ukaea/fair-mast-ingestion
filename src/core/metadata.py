from typing import Optional

import xarray as xr
from distributed import Lock
from pydantic import BaseModel
from sqlalchemy import (
    Column,
    Integer,
    MetaData,
    String,
    Table,
    and_,
    create_engine,
    delete,
)
from sqlalchemy.dialects.sqlite import insert as sqlite_upsert

from src.core.log import logger
from src.core.utils import connected_to_cluster, get_uuid, nullcontext


class SourceMetadata(BaseModel):
    uuid: str
    name: str
    shot_id: int
    url: str
    description: Optional[str] = ""
    quality: Optional[str] = "Not Checked"
    imas: Optional[str] = None

    class Config:
        extra = "ignore"


class SignalMetadata(BaseModel):
    uuid: str
    name: str
    source: str
    shot_id: int
    url: str
    rank: int
    shape: str
    dimensions: str
    units: Optional[str] = ""
    description: Optional[str] = ""
    quality: Optional[str] = "Not Checked"
    imas: Optional[str] = None

    class Config:
        extra = "ignore"


class MetadataWriter:
    def __init__(self, uri: str, remote_path: str):
        self.engine = create_engine(uri)
        self.data = []
        self.remote_path = remote_path

        with self._get_lock():
            self.signals_table = self.create_signals_table()
            self.sources_table = self.create_sources_table()

    def create_sources_table(self) -> Table:
        metadata = MetaData()
        table = Table(
            "sources",
            metadata,
            Column("uuid", String, primary_key=True),
            Column("shot_id", Integer),
            Column("name", String),
            Column("url", String),
            Column("description", String),
            Column("quality", String),
            Column("imas", String),
        )
        metadata.create_all(self.engine)
        return table

    def create_signals_table(self) -> Table:
        metadata = MetaData()
        table = Table(
            "signals",
            metadata,
            Column("uuid", String, primary_key=True),
            Column("shot_id", Integer),
            Column("name", String),
            Column("source", String),
            Column("url", String),
            Column("description", String),
            Column("units", String),
            Column("quality", String),
            Column("rank", Integer),
            Column("shape", String),
            Column("dimensions", String),
            Column("imas", String),
        )
        metadata.create_all(self.engine)
        return table

    def write(self, shot: int, dataset: xr.Dataset):
        lock = self._get_lock()

        with lock:
            self.write_source(shot, dataset)
            self.write_signals(shot, dataset)

    def write_source(self, shot: int, dataset: xr.Dataset):
        name = dataset.attrs["name"]
        url = f"{self.remote_path}/{shot}.zarr/{name}"
        data = SourceMetadata(
            uuid=get_uuid(name, shot),
            shot_id=shot,
            url=url,
            **dataset.attrs,
        )
        data = data.model_dump()

        self._delete_sources(data["name"], shot)
        self._perform_upsert(self.sources_table, [data])

    def write_signals(self, shot: int, dataset: xr.Dataset):
        datas = []

        source_name = dataset.attrs["name"]

        for item in dataset.data_vars.values():
            full_name = f"{source_name}/{item.attrs['name']}"
            url = f"{self.remote_path}/{shot}.zarr/{full_name}"

            rank = len(item.shape)
            shape = ",".join(list(map(str, item.shape)))
            dims = ",".join(list(item.sizes.keys()))

            data = SignalMetadata(
                uuid=get_uuid(full_name, shot),
                shot_id=shot,
                url=url,
                source=source_name,
                shape=shape,
                dimensions=dims,
                rank=rank,
                **item.attrs,
            )
            datas.append(data.model_dump())

        if len(datas) == 0:
            logger.warning(f"No signals found for shot {shot} and source {source_name}")
            return

        self._delete_signals(source_name, shot)
        self._perform_upsert(self.signals_table, datas)

    def _get_lock(self):
        if connected_to_cluster():
            lock = Lock("database-lock")
            logger.debug("Connected to cluster. Locking database.")
        else:
            lock = nullcontext()

        return lock

    def _delete_sources(self, name: str, shot: int):
        stmt = delete(self.sources_table).where(
            and_(
                self.sources_table.c.shot_id == shot,
                self.sources_table.c.name == name,
            )
        )

        with self.engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

    def _delete_signals(self, name: str, shot: int):
        stmt = delete(self.signals_table).where(
            and_(
                self.signals_table.c.shot_id == shot,
                self.signals_table.c.source == name,
            )
        )

        with self.engine.connect() as conn:
            conn.execute(stmt)
            conn.commit()

    def _perform_upsert(self, table: Table, data: list[dict]):
        # Insert or update operation
        # Create an insert statement with upsert behavior
        stmt = sqlite_upsert(table).values(data)
        upsert_stmt = stmt.on_conflict_do_update(
            index_elements=["uuid"],
            set_={
                key: stmt.excluded[key] for key in data[0] if key != "id"
            },  # Update with new values on conflict
        )

        with self.engine.connect() as conn:
            conn.execute(upsert_stmt)
            conn.commit()  # Commit the transaction if required
