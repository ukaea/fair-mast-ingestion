from src.level2.reader import DatasetReader
from src.level2.api import create_reader


def test_create_reader_with_mapping(sample_mapping):
    reader = create_reader(sample_mapping, "uda")
    assert isinstance(reader, DatasetReader)


def test_create_reader_with_mapping_name():
    reader = create_reader("mast", "uda")
    assert isinstance(reader, DatasetReader)


def test_create_reader_with_mapping_file():
    reader = create_reader("src/mappings/mast.yml", "uda")
    assert isinstance(reader, DatasetReader)
