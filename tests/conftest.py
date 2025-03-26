import numpy as np
import pytest
import xarray as xr

from src.core.model import DatasetInfo, Mapping, ProfileInfo


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    dataset = xr.Dataset(
        data_vars=dict(
            data=("time", np.random.random(100)),
            error=("time", np.random.random(100)),
            time=("time", np.linspace(0, 10, 100)),
        )
    )
    return dataset


@pytest.fixture
def sample_dataarray() -> xr.DataArray:
    return xr.DataArray(np.random.random(100), dims=["time"])


@pytest.fixture
def sample_mapping() -> Mapping:
    mapping = Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="magnetics/ip",
        datasets={
            "magnetics": DatasetInfo(
                profiles={"ip": ProfileInfo(source="ip", dimensions={"time": None})}
            )
        },
    )
    return mapping


@pytest.fixture
def fake_dataset():
    return xr.Dataset(
        data_vars=dict(
            data=("time", np.random.random(10)),
            time=("time", np.random.random(10)),
            error=("time", np.random.random(10)),
        ),
        attrs={"name": "plasma_current", "shot_id": 30420, "source": "amc"},
    )


@pytest.fixture
def fake_image():
    return xr.Dataset(
        data_vars=dict(
            data=(("frames", "heigth", "width"), np.random.random((20, 10, 10))),
        ),
        attrs={"name": "rbb", "shot_id": 30420, "source": "rbb"},
    )


@pytest.fixture
def fake_channel_dataset(fake_dataset):
    channels = {f"channel{i}": fake_dataset["data"] for i in range(10)}
    for channel in channels.values():
        channel.attrs.update(fake_dataset.attrs)
    return xr.Dataset(channels)
