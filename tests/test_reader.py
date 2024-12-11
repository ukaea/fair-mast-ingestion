import importlib
from dataclasses import asdict

import pandas as pd
import pytest
import xarray as xr

from src.reader import (  # noqa: E402
    DatasetReader,  # noqa: E402
    SignalMetadataReader,  # noqa: E402
    SourceMetadataReader,  # noqa: E402
)  # noqa: E402

uda_available = not importlib.util.find_spec("pyuda")


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_list_signals():
    shot = 30420
    reader = DatasetReader(shot)
    signals = reader.list_datasets()

    assert isinstance(signals, list)
    assert len(signals) == 11254

    info = signals[0]
    assert info.name == "abm/calib_shot"


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_list_signals_exclude_raw():
    shot = 30420
    reader = DatasetReader(shot)
    signals = reader.list_datasets(exclude_raw=True)

    assert isinstance(signals, list)
    assert len(signals) == 890

    info = signals[0]
    assert info.name == "abm/calib_shot"


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_read_signal():
    shot = 30420
    reader = DatasetReader(shot)
    signals = reader.list_datasets()
    info = asdict(signals[0])
    info["format"] = "IDA"
    dataset = reader.read_dataset(info)

    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs["name"] == "abm/calib_shot"
    assert dataset["time"].shape == (1,)


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_read_image():
    shot = 30420
    reader = DatasetReader(shot)

    signals = reader.list_datasets()
    signals = filter(lambda x: x.signal_type == "Image", signals)
    signals = list(signals)

    dataset = reader.read_dataset(asdict(signals[0]))

    assert isinstance(dataset, xr.Dataset)
    assert dataset.attrs["name"] == "rba"
    assert dataset["time"].shape == (186,)
    assert dataset["data"].shape == (186, 912, 768)
    assert list(dataset.dims.keys()) == ["time", "height", "width"]


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_read_signals_metadata():
    shot = 30420
    reader = SignalMetadataReader(shot)
    df = reader.read_metadata()

    assert isinstance(df, pd.DataFrame)


@pytest.mark.skipif(uda_available, reason="Pyuda client unavailable")
def test_read_sources_metadata():
    shot = 30420
    reader = SourceMetadataReader(shot)
    df = reader.read_metadata()

    assert isinstance(df, pd.DataFrame)
