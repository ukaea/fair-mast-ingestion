import json
from pathlib import Path

import numpy as np

FIXTURES_DIR = Path(__file__).parent / "fixtures" / "geometry"


class FakeUdaNode:
    """Stands in for a pyuda ``StructuredWritable`` node.

    Real geometry parsing needs two different things from this object
    depending on which profile is being processed: the raw JSON tree
    (``.jsonify()``) and/or specific numpy arrays fetched by attribute
    name (e.g. ``node.R``). Both are backed by data captured from a
    real UDA server (see tests/level2/fixtures/geometry) rather than
    hand-built, so the parsing logic under test runs against realistic
    shapes/dtypes.
    """

    def __init__(self, raw_json=None, arrays=None):
        self._raw_json = raw_json
        self._arrays = arrays or {}

    def jsonify(self):
        return self._raw_json

    def __getattr__(self, name):
        try:
            return self._arrays[name]
        except KeyError:
            raise AttributeError(name)


class FakeGeometryTree:
    """Stands in for the object returned by ``pyuda.Client.geometry(...)``."""

    def __init__(self, stem, raw_json, extra_arrays=None):
        self.data = {stem: FakeUdaNode(raw_json=raw_json)}
        for key, attrs in (extra_arrays or {}).items():
            self.data[key] = FakeUdaNode(arrays=attrs)


def load_geometry_fixture(name: str):
    """Load a real UDA geometry capture for a given fixture name.

    Returns (raw_json_string, metadata_dict, extra_arrays) where
    extra_arrays maps the extra ``geom_data.data[key]`` lookups some
    profile types perform (saddle/pf/xray/limiter) to the real numpy
    arrays a live server returned for them.
    """
    raw_json = (FIXTURES_DIR / f"{name}.json").read_text()
    meta = json.loads((FIXTURES_DIR / f"{name}_meta.json").read_text())

    extra_arrays = {}
    manifest_path = FIXTURES_DIR / f"geom_access_{name}_manifest.json"
    npz_path = FIXTURES_DIR / f"geom_access_{name}.npz"
    if manifest_path.exists() and npz_path.exists():
        manifest = json.loads(manifest_path.read_text())
        npz = np.load(npz_path)
        for key, attr_names in manifest.items():
            extra_arrays[key] = {attr: npz[f"{key}::{attr}"] for attr in attr_names}

    return raw_json, meta, extra_arrays


def patch_uda_geometry(mocker, stem: str, fixture_name: str):
    """Patch the module-level UDA geometry fetch functions to serve a
    real, pre-captured fixture instead of hitting a live UDA server."""
    raw_json, meta, extra_arrays = load_geometry_fixture(fixture_name)
    tree = FakeGeometryTree(stem, raw_json, extra_arrays)

    mocker.patch("src.core.load._fetch_uda_geometry_tree", return_value=tree)
    mock_meta = mocker.patch("src.core.load._fetch_uda_geom_metadata")
    mock_meta.return_value = meta
    return tree
