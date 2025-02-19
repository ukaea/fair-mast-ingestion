import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import fnmatch

def create_header(ncfile, headerdict):
    for key, value in headerdict.items():
        setattr(ncfile, key, value)

def create_saddle_variable(group, parquet_file, var_type, version):
    """Create x-ray camera variables and add to group. Use XSX parquet files."""
    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))

        data = np.empty(1, var_type.dtype_view)
        data["name"][:] = row["uda_name"].replace("/", "_")
        data["version"] = version
        data["coordinate"]["r1"] = row["r1"]
        data["coordinate"]["r2"] = row["r2"]
        data["coordinate"]["z1"] = row["z1"]
        data["coordinate"]["z2"] = row["z2"]
        data["coordinate"]["angle"] = row["theta"] # unsure if a geometry thing
        data["geometry"]["p"] = row["p"] # what is this

        var[:] = data
        #var.setnccattr("units", "SI units: degrees, m")

def xsx_parquet_to_netcdf(netcdf_file, headerdict):
    """Convert parquet file to netcdf."""
    with Dataset(netcdf_file, "w", format="NETCDF4") as ncfile:

        """Add global attributes."""
        create_header(ncfile, headerdict)

        """Create X-ray camera group."""
        camera_group = ncfile.createGroup("cameragroup")

        """Group by horizontal, vertical and tangential cameras."""
        horizontal_group = camera_group.createGroup("horizontal")
        vertical_group = camera_group.createGroup("vertical")
        tangential_group = camera_group.createGroup("tangential")

        """Define data types and combine into compound data types."""
        coord_dtype = np.dtype([("r1", "<f8"), ("z1", "<f8"), ("r2", "<f8"), ("z2", "<f8"), ("theta", "<f8")])
        geometry_dtype = np.dtype([
            ("p", "<f8"),
            # corner radius
            # area
            # angle
        ])

        camera_dtype = np.dtype([
            ("name", "S10"),
            ("version", "<f8"),
            ("coordinate", coord_dtype),
            ("geometry", geometry_dtype)
        ])

        camera_group.createCompoundType(coord_dtype, "COORDINATES")
        camera_group.createCompoundType(geometry_dtype, "GEOMETRY")
        
        var_type = camera_group.createCompoundType(camera_dtype, "XRAYCAM")

        camera_group.createDimension("singleDim", 1)

        """Assign parquet file contents to appropriate groups."""
        parquet_files = {
            "inner_vertical": ("geometry/data/xsx/ssx_inner_vertical_cam.parquet", "INNER_VERTICAL"),
            "lower_horizontal": ("geometry/data/xsx/ssx_lower_horizontal_cam.parquet", "LOWER_HORIZONTAL"),
            "outer_vertical": ("geometry/data/xsx/ssx_outer_vertical_cam.parquet", "OUTER_VERTICAL"),
            "tangential": ("geometry/data/xsx/ssx_tangential_cam.parquet", "TANGENTIAL"),
            "third_horizontal": ("geometry/data/xsx/ssx_third_horizontal_cam.parquet", "THIRD_HORIZONTAL"),
            "upper_horizontal": ("geometry/data/xsx/ssx_upper_horizontal_cam.parquet", "UPPER_HORIZONTAL")
        }

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        patterns = ["*vertical", "*horizontal", "tangential"]
        groups = [vertical_group, horizontal_group, tangential_group]

        keys = list(parquet_files.keys())
        
        for i in range(0,len(patterns)): # indexing through patterns
            
            matching_keys = fnmatch.filter(keys, patterns[i]) # outputs list of pq file keys that match a pattern
            matching_values = list(dict((k, parquet_files[k]) for k in matching_keys).values()) # gets filepaths corresponding to keys that match the pattern
            matching_filepaths = []

            for q in matching_values:
                matching_filepaths.append(q[0])
            
            # print(matching_filepaths)
            for n in matching_filepaths:
                create_saddle_variable(groups[i], n, var_type, version) # add variable to matching group



if __name__ == "__main__":
    # Metadata for the NetCDF file
    headerdict = {
    "Conventions": "",
    "device": "MAST",
    "class": "magnetics",
    "system": "xraycameras",
    "configuration": "geometry",
    "shotRangeStart": 0,
    "shotRangeStop": 400000,
    "content": "geometry of the X-ray cameras for MAST",
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
    "creatorCode": "python create_netcdf_xraycams.py",
    "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
    "createdBy": "sfrankel",
    "testCode": "",
    "testDate": "",
    "testedBy": "",
    }
        
    xsx_parquet_to_netcdf("geometry/xraycams.nc", headerdict)