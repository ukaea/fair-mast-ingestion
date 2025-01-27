import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

def create_header(ncfile, headerdict):
    for key, value in headerdict.items():
        setattr(ncfile, key, value)

def create_fluxloop_variable(group, parquet_file, lp_cp, location, version):
    df = pd.read_parquet(parquet_file)
    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"].replace("/", "_"), lp_cp, ("singleDim",))

        # Initialize data
        data = np.empty(1, lp_cp.dtype)
        data["name"][:] = row["uda_name"].replace("/", "_")
        data["version"] = version
        data["location"] = location
        data["coordinate"]["r"] = row["r"]
        data["coordinate"]["z"] = row["z"]
        data['geometry']['phi_start'] = np.nan
        data['geometry']['phi_end'] = np.nan

        # Assign data to variable
        var[:] = data
        var.setncattr("units", "SI units: degrees, m")

def parquet_to_netcdf(netcdf_file, headerdict):
    """Convert Parquet files to a NetCDF structure."""
    with Dataset(netcdf_file, 'w', format='NETCDF4') as ncfile:
        # Attach metadata
        create_header(ncfile, headerdict)

        # Create the magnetics group
        mag_grp = ncfile.createGroup("magnetics")
        fluxloop_group = mag_grp.createGroup("fluxloops")

        # Create fluxloop subgroups
        fluxloop_subgroups = {
            "p2_lower": fluxloop_group.createGroup("p2/lower"),
            "p2_upper": fluxloop_group.createGroup("p2/upper"),
            "p3_lower": fluxloop_group.createGroup("p3/lower"),
            "p3_upper": fluxloop_group.createGroup("p3/upper"),
            "p4_lower": fluxloop_group.createGroup("p4/lower"),
            "p4_upper": fluxloop_group.createGroup("p4/upper"),
            "p5_lower": fluxloop_group.createGroup("p5/lower"),
            "p5_upper": fluxloop_group.createGroup("p5/upper"),
            "p6_lower": fluxloop_group.createGroup("p6/lower"),
        }

        # Define data types
        coord_dtype = np.dtype([("r", "<f8"), ("z", "<f8")])
        geometry_dtype = np.dtype([("phi_start", "<f4"), ("phi_end", "<f4")])
        fl_dtype = np.dtype([
            ("name", "S50"),
            ("version", "<f8"),
            ("location", "S50"),
            ("coordinate", coord_dtype),
            ("geometry", geometry_dtype),
        ])

        # Register compound types
        fluxloop_group.createCompoundType(coord_dtype, "COORDINATE")
        fluxloop_group.createCompoundType(geometry_dtype, "GEOMETRY")
        lp_cp = fluxloop_group.createCompoundType(fl_dtype, "FLUXLOOP")

        # Create a shared dimension
        fluxloop_group.createDimension("singleDim", 1)

        # Define Parquet file paths and metadata
        parquet_files = {
            "p2_lower": ("geometry/data/amb/fl_p2l.parquet", "P2 lower"),
            "p2_upper": ("geometry/data/amb/fl_p2u.parquet", "P2 upper"),
            "p3_lower": ("geometry/data/amb/fl_p3l.parquet", "P3 lower"),
            "p3_upper": ("geometry/data/amb/fl_p3u.parquet", "P3 upper"),
            "p4_lower": ("geometry/data/amb/fl_p4l.parquet", "P4 lower"),
            "p4_upper": ("geometry/data/amb/fl_p4u.parquet", "P4 upper"),
            "p5_lower": ("geometry/data/amb/fl_p5l.parquet", "P5 lower"),
            "p5_upper": ("geometry/data/amb/fl_p5u.parquet", "P5 upper"),
            "p6_lower": ("geometry/data/amb/fl_p6l.parquet", "P6 lower"),
        }

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        create_fluxloop_variable()

        # Process each parquet file
        for subgroup_key, (file_path, location) in parquet_files.items():
            create_fluxloop_variable(
                fluxloop_subgroups[subgroup_key], file_path, lp_cp, location, version
            )

if __name__ == "__main__":
    # Metadata for the NetCDF file
    headerdict = {
        "Conventions": "",
        "device": "MAST",
        "class": "magnetics",
        "system": "fluxloops",
        "configuration": "geometry",
        "shotRangeStart": 0,
        "shotRangeStop": 400000,
        "content": "geometry of the fluxloops for MAST",
        "comment": "",
        "units": "SI, degrees, m",
        "coordinateSystem": "Cylindrical",
        "structureCastType": "unknown",
        "calibration": "None",
        "version": 0,
        "revision": 0,
        "status": "development",
        "releaseDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
        "releaseTime": datetime.strftime(datetime.now(), "%H:%M:%S"),
        "owner": "jhodson",
        "signedOffBy": "",
        "signedOffDate": "",
        "creatorCode": "python create_netcdf_fluxloops.py",
        "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
        "createdBy": "jhodson",
        "testCode": "",
        "testDate": "",
        "testedBy": "",
    }

    parquet_to_netcdf("test.nc", headerdict)
