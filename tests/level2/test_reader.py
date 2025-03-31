from src.core.load import UDALoader
from src.level2.reader import DatasetReader


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
        background_window=mocker.Mock(tmin=0, tmax=5)
    ))
    shot = 30420
    loader = UDALoader()
    reader = DatasetReader(sample_mapping, loader)
    profile = reader.read_profile(shot, "magnetics", "ip")
    background_mean = sample_dataarray.isel(time=slice(0, 5)).mean(dim="time")
    expected = sample_dataarray - background_mean
    assert (profile == expected).all()