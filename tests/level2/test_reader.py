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
