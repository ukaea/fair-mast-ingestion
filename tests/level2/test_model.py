from src.core.model import load_model, Mapping


def test_load_mappings():
    load_model("src/mappings/mast.yml")
    load_model("src/mappings/mastu.yml")
    load_model("src/mappings/jet.yml")


def test_load_model():
    config = load_model("src/mappings/mast_s3.yml")
    assert isinstance(config, Mapping)
