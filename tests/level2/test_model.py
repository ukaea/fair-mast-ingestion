from src.core.model import Mapping, load_model


def test_load_mappings():
    load_model("src/level2/mappings/mast.yml")
    load_model("src/level2/mappings/mastu.yml")
    load_model("src/level2/mappings/jet.yml")


def test_load_model():
    config = load_model("src/level2/mappings/mast_s3.yml")
    assert isinstance(config, Mapping)
