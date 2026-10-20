import functools
import os
import tomllib
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from src.core.model import DatasetInfo, Mapping, ProfileInfo

REFERENCE_SHOT = 30421


def _apply_pytest_env():
    """Set the UDA variables from pyproject, so pytest-env is optional.

    `[tool.pytest_env]` only takes effect when pytest-env is installed, which it
    is in CI but often not locally. Reading it here keeps one source of truth
    and lets the UDA-marked tests run from a laptop on the VPN.
    """
    pyproject = Path(__file__).resolve().parent.parent / "pyproject.toml"
    try:
        env = tomllib.loads(pyproject.read_text()).get("tool", {}).get("pytest_env", {})
    except (OSError, tomllib.TOMLDecodeError):
        return
    for key, value in env.items():
        os.environ.setdefault(key, str(value))


_apply_pytest_env()


@functools.lru_cache(maxsize=1)
def uda_available() -> bool:
    """Is a UDA server reachable? Checked once, lazily, never at import time."""
    if os.environ.get("CI"):
        return False
    try:
        import pyuda

        pyuda.Client().get("ip", REFERENCE_SHOT)
        return True
    except Exception:
        return False


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "uda: needs a live UDA connection; skipped in CI and off the VPN",
    )


def pytest_collection_modifyitems(config, items):
    if uda_available():
        return
    skip = pytest.mark.skip(reason="no UDA connection (set CI= and join the VPN)")
    for item in items:
        if "uda" in item.keywords:
            item.add_marker(skip)


@pytest.fixture(scope="session")
def reference_shot() -> int:
    return REFERENCE_SHOT


@pytest.fixture
def sample_dataset() -> xr.Dataset:
    dataset = xr.Dataset(
        data_vars=dict(
            data=("time", np.random.random(100)),
            error=("time", np.random.random(100)),
            time=("time", np.linspace(-5, 10, 100)),
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
