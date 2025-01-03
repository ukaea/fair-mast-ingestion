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
        datasets={
            "magnetics": DatasetInfo(
                profiles={"ip": ProfileInfo(source="ip", dimensions={"time": None})}
            )
        },
    )
    return mapping
