import xarray as xr

from src.core.metadata import ParquetMetadataWriter


def test_parquet_signal_metadata_uses_each_signals_attributes(tmp_path):
    dataset = xr.Dataset(
        data_vars={
            "plasma_current": xr.DataArray(
                [1.0, 2.0],
                dims=["time"],
                attrs={
                    "units": "A",
                    "description": "Plasma current",
                    "quality": "Checked",
                    "imas": "magnetics.ip.0",
                },
            ),
            "neutron_rate": xr.DataArray(
                [3.0, 4.0],
                dims=["time"],
                attrs={
                    "units": "Hz",
                    "description": "Total neutron rate",
                    "quality": "Validated",
                    "imas": "summary.fusion.neutron_rates.total",
                },
            ),
        },
        attrs={
            "name": "summary",
            "units": "group units",
            "description": "Group description",
            "quality": "Group quality",
            "imas": "summary",
        },
    )
    writer = ParquetMetadataWriter(tmp_path, "s3://example")

    signals = writer.write_signals(30420, dataset)

    metadata_by_name = {signal["name"]: signal for signal in signals}
    assert {
        field: metadata_by_name["plasma_current"][field]
        for field in ("units", "description", "quality", "imas")
    } == {
        "units": "A",
        "description": "Plasma current",
        "quality": "Checked",
        "imas": "magnetics.ip.0",
    }
    assert {
        field: metadata_by_name["neutron_rate"][field]
        for field in ("units", "description", "quality", "imas")
    } == {
        "units": "Hz",
        "description": "Total neutron rate",
        "quality": "Validated",
        "imas": "summary.fusion.neutron_rates.total",
    }
