from pathlib import Path

import pytest

from src.core.model import DatasetInfo, Mapping, ProfileInfo, load_model

MAPPING_DIR = Path("mappings/level2")
MAPPING_FILES = sorted(MAPPING_DIR.glob("*.yml"))


def _mapping_id(path: Path) -> str:
    return path.stem


def test_mapping_files_exist():
    # Guard against the glob silently matching nothing (e.g. wrong cwd),
    # which would make the parametrized tests below trivially pass.
    assert MAPPING_FILES, f"No mapping files found in {MAPPING_DIR}"


@pytest.mark.parametrize("mapping_file", MAPPING_FILES, ids=_mapping_id)
def test_mapping_file_parses(mapping_file: Path):
    mapping = load_model(mapping_file)
    assert isinstance(mapping, Mapping)


@pytest.mark.parametrize("mapping_file", MAPPING_FILES, ids=_mapping_id)
def test_mapping_file_structure(mapping_file: Path):
    mapping = load_model(mapping_file)

    assert mapping.facility, "facility must be a non-empty string"
    assert mapping.default_loader, "default_loader must be a non-empty string"
    assert mapping.plasma_current, "plasma_current must be a non-empty string"
    assert mapping.datasets, "mapping must define at least one dataset"

    for name, dataset in mapping.datasets.items():
        assert isinstance(dataset, DatasetInfo)
        assert dataset.profiles, f"dataset '{name}' must define at least one profile"

        for profile_name, profile in dataset.profiles.items():
            qualified = f"{name}.{profile_name}"
            assert isinstance(profile, ProfileInfo)
            # A profile is sourced either from a signal (``source``) or from a
            # geometry block; at least one of the two must be present.
            assert profile.source or profile.geometry is not None, (
                f"profile '{qualified}' must define a source or geometry"
            )
            assert profile.dimensions, (
                f"profile '{qualified}' must define at least one dimension"
            )

            if profile.dimension_order is not None:
                missing = set(profile.dimension_order) - set(profile.dimensions)
                assert not missing, (
                    f"profile '{qualified}' has dimension_order entries "
                    f"{sorted(missing)} not present in its dimensions"
                )
