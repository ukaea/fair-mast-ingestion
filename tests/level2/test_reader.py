import numpy as np
import xarray as xr

from src.core.load import UDALoader
from src.core.model import Mapping
from src.level2.reader import DatasetReader
from src.level2.transforms import DatasetInterpolationTransform, InterpolationParams


def test_read_profiles(mocker, sample_mapping, sample_dataarray):
    mocker.patch("src.core.load.UDALoader.load", return_value=sample_dataarray)
    shot = 30420

    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profiles = reader.read_profiles(shot, "magnetics")

    assert (profiles["ip"] == sample_dataarray).all()


def test_read_profile(mocker, sample_mapping, sample_dataarray):
    mocker.patch("src.core.load.UDALoader.load", return_value=sample_dataarray)
    shot = 30420
    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profiles = reader.read_profile(shot, "magnetics", "ip")
    assert (profiles == sample_dataarray).all()


def test_read_profile_with_background_subtraction(mocker, sample_mapping, sample_dataarray):
    mocker.patch("src.core.load.UDALoader.load", return_value=sample_dataarray)
    mocker.patch("src.level2.reader.DatasetReader._get_source", return_value=mocker.Mock(
        background_correction=mocker.Mock(tmin=0, tmax=5)
    ))
    shot = 30420
    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profile = reader.read_profile(shot, "magnetics", "ip")
    background_mean = sample_dataarray.isel(time=slice(0, 5)).mean(dim="time")
    expected = sample_dataarray - background_mean
    assert (profile == expected).all()

def test_interpolation_aligns_grids_across_shots():
    mapping = Mapping(
        facility="MAST", default_loader="uda",
        plasma_current="summary/ip", datasets={},
    )
    transform = DatasetInterpolationTransform(None, mapping)
    params = InterpolationParams(start=-0.1, step=2.5e-4, method="zero")

    t_a = np.arange(-0.05, 0.10, 1.5e-3)
    t_b = np.arange(-0.06, 0.20, 1.3e-3)
    a = xr.DataArray(np.ones(len(t_a)), dims=["time"], coords={"time": t_a}, name="x").to_dataset()
    b = xr.DataArray(np.ones(len(t_b)), dims=["time"], coords={"time": t_b}, name="x").to_dataset()

    a_out = transform.interpolate_dimension(a, "time", params)
    b_out = transform.interpolate_dimension(b, "time", params)

    n = min(len(a_out.time), len(b_out.time))
    assert np.allclose(a_out.time.values[:n], b_out.time.values[:n])