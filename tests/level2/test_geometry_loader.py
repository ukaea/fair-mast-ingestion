"""Tests for Level2UDAGeometryLoader.

Level2 profiles backed by ``geometry:`` blocks in the mapping files never
go through the standard signal loader, so they were previously untested
outside of a live UDA connection. The fixtures under
tests/level2/fixtures/geometry/ are real UDA responses captured once
(see git history) so these tests exercise the exact parsing branches
(default/saddle/pf-coil/x-ray-cam/limiter) with realistic shapes, dtypes
and metadata, entirely offline.
"""

from types import SimpleNamespace

import numpy as np
import xarray as xr

from src.core.load import Level2UDAGeometryLoader
from tests.level2.conftest import patch_uda_geometry


def _geometry(**kwargs):
    defaults = dict(
        stem=None,
        path=None,
        shot=None,
        measurement=None,
        channel_name="geometry_channel",
    )
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_default_geometry_profile(mocker):
    """No special-cased profile name: plain scalar-per-row geometry (botcol)."""
    patch_uda_geometry(mocker, stem="centrecolumn/botcol", fixture_name="botcol")

    geometry = _geometry(
        stem="centrecolumn/botcol",
        path="/passive/efit",
        shot="/common/uda-scratch/jg3176/passivestructures.nc",
        measurement="centreR",
        channel_name="botcol_geometry_channel",
    )

    loader = Level2UDAGeometryLoader()
    result = loader.run(geometry, "botcol_r")

    assert isinstance(result, xr.DataArray)
    assert result.dims == ("botcol_geometry_channel",)
    assert result.shape == (1,)
    np.testing.assert_allclose(result.values, [0.23537670075893402])
    assert list(result.coords["botcol_geometry_channel"].values) == ["botcol"]

    # Metadata is decoded from base64-encoded int64 arrays into plain scalars.
    assert result.attrs["device"] == "MAST"
    assert result.attrs["shotRangeStart"] == 0
    assert result.attrs["shotRangeStop"] == 40000


def test_saddle_geometry_profile(mocker):
    """'b_field_tor_probe_saddle' profiles re-fetch r/z/phi coil paths per channel."""
    patch_uda_geometry(mocker, stem="lower", fixture_name="saddle")

    geometry = _geometry(
        stem="lower",
        path="/magnetics/saddlecoils",
        shot="/common/uda-scratch/jg3176/saddle.nc",
        measurement="r",
        channel_name="b_field_tor_probe_saddle_l_geometry_channel",
    )

    loader = Level2UDAGeometryLoader()
    result = loader.run(geometry, "b_field_tor_probe_saddle_l_r")

    assert result.dims == ("b_field_tor_probe_saddle_l_geometry_channel", "coordinate")
    assert result.shape == (12, 28)
    assert result.coords["coordinate"].shape == (28,)
    assert (
        list(result.coords["b_field_tor_probe_saddle_l_geometry_channel"].values)[0]
        == "sad_out_l01"
    )


def test_pf_coil_geometry_profile(mocker):
    """PF coil profiles ('p2_inner' etc.) pull array data from geom_elements."""
    patch_uda_geometry(mocker, stem="p2/p2_inner_upper", fixture_name="pf")

    geometry = _geometry(
        stem="p2/p2_inner_upper",
        path="/magnetics/pfcoil",
        shot="/common/uda-scratch/jg3176/pfcoils.nc",
        measurement="centreR",
        channel_name="p2_inner_upper_coordinate_element",
    )

    loader = Level2UDAGeometryLoader()
    result = loader.run(geometry, "p2_inner_upper_r")

    assert result.dims == ("p2_inner_upper_coordinate_element",)
    assert result.shape == (12,)
    assert result.dtype == np.float32


def test_xray_cam_geometry_profile(mocker):
    """X-ray camera ('cam' in name) profiles pull origin/endpoint arrays."""
    patch_uda_geometry(mocker, stem="tangential", fixture_name="xray")

    geometry = _geometry(
        stem="tangential",
        path="/xraycams/core",
        shot="/common/uda-scratch/jg3176/xraycams.nc",
        measurement="origin_r",
        channel_name="tangential_cam_geometry_channel",
    )

    loader = Level2UDAGeometryLoader()
    result = loader.run(geometry, "tangential_cam_origin_r")

    assert result.dims == ("tangential_cam_geometry_channel",)
    assert result.shape == (18,)


def test_limiter_geometry_profile(mocker):
    """Limiter profiles pull R/Z arrays straight from efit/data."""
    patch_uda_geometry(mocker, stem="efit", fixture_name="limiter")

    geometry = _geometry(
        stem="efit",
        path="/limiter",
        shot="/common/uda-scratch/jg3176/limiter.nc",
        measurement="R",
        channel_name="limiter_geometry_channel",
    )

    loader = Level2UDAGeometryLoader()
    result = loader.run(geometry, "limiter_r")

    assert result.dims == ("limiter_geometry_channel",)
    assert result.shape == (37,)
    assert result.dtype == np.float32
    assert list(result.coords["limiter_geometry_channel"].values[:2]) == [
        "element_1",
        "element_2",
    ]


def test_geometry_profile_sets_imas_and_description(mocker):
    patch_uda_geometry(mocker, stem="efit", fixture_name="limiter")

    from src.core.model import DatasetInfo, Geometry, Mapping, ProfileInfo
    from src.level2.reader import DatasetReader

    geometry = Geometry(
        stem="efit",
        path="/limiter",
        shot="/common/uda-scratch/jg3176/limiter.nc",
        measurement="R",
        channel_name="limiter_geometry_channel",
    )

    mapping = Mapping(
        facility="MAST",
        default_loader="uda",
        plasma_current="wall/limiter_r",
        datasets={
            "wall": DatasetInfo(
                profiles={
                    "limiter_r": ProfileInfo(
                        source="",
                        geometry=geometry,
                        dimensions={"limiter_geometry_channel": None},
                        description="Major radius values of the limiter",
                    )
                }
            )
        },
    )

    reader = DatasetReader(mapping, loader=mocker.Mock())
    profile_info = mapping.datasets["wall"].profiles["limiter_r"]
    result = reader.read_geometry(profile_info, "limiter_r")

    assert result.attrs["imas"] == ""
    assert result.attrs["description"] == "Major radius values of the limiter"
