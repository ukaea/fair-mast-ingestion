from src.core.model import Mapping, load_model


def test_load_mappings():
    load_model("mappings/level2/mast.yml")
    load_model("mappings/level2/mastu.yml")
    load_model("mappings/level2/jet.yml")


def test_load_model():
    config = load_model("mappings/level2/mast_s3.yml")
    assert isinstance(config, Mapping)
