import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import fnmatch

def create_header(ncfile, headerdict):
    for key, value in headerdict.items():
        setattr(ncfile, key, value)

def create_saddle_variable(group, parquet_file, var_type, version):
    """Create saddle coil variables and add to group. Use XMB parquet files."""
    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))

        data = np.empty(1, var_type.dtype_view)
        data["name"][:] = row["uda_name"].replace("/", "_")
        data["version"] = version
        data["sector"] = row["sector"]
        data["coordinate"]["r"] = row["r"]
        data["coordinate"]["z"] = row["z"]
        data["coordinate"]["phi"] = row["toroidal_angle"]
        data["geometry"]["height"] = row["height"]
        data["geometry"]["width"] = row["width"]
        # missing: curved vs flat, coilPath, corner radius

        var[:] = data
        #var.setnccattr("units", "SI units: degrees, m")

def xmb_parquet_to_netcdf(netcdf_file, headerdict):
    """Convert parquet file to netcdf."""
    with Dataset(netcdf_file, "w", format="NETCDF4") as ncfile:

        """Add global attributes."""
        create_header(ncfile, headerdict)

        """Create saddle coils group."""
        saddle_group = ncfile.createGroup("saddlegroup")

        """Group by lower, middle and upper saddle coils."""
        lower_group = saddle_group.createGroup("lower")
        middle_group = saddle_group.createGroup("middle")
        upper_group = saddle_group.createGroup("upper")

        """Define data types and combine into compound data types."""
        coord_dtype = np.dtype([("r", "<f8"), ("z", "<f8"), ("phi", "<f8")])
        geometry_dtype = np.dtype([
            ("height", "<f8"),
            ("width", "<f8"),
            # corner radius
            # area
            # angle
        ])

        saddle_dtype = np.dtype([
            ("name", "S10"),
            ("version", "<f8"),
            ("sector", "<f8"),
            ("coordinate", coord_dtype),
            ("geometry", geometry_dtype)
        ])

        saddle_group.createCompoundType(coord_dtype, "COORDINATES")
        saddle_group.createCompoundType(geometry_dtype, "GEOMETRY")
        
        var_type = saddle_group.createCompoundType(saddle_dtype, "SADDLECOIL")

        saddle_group.createDimension("singleDim", 1)

        """Assign parquet file contents to appropriate groups."""
        parquet_files = {
            "lower_saddles": ("geometry/data/xmb/xmb_sad_l.parquet", "LOWER"),
            "middle_saddles": ("geometry/data/xmb/xmb_sad_m.parquet", "MIDDLE"),
            "upper_saddles": ("geometry/data/xmb/xmb_sad_u.parquet", "UPPER")
        }

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        patterns = ["lower*", "middle*", "upper*"]
        groups = [lower_group, middle_group, upper_group]

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
    "system": "saddlecoils",
    "configuration": "geometry",
    "shotRangeStart": 0,
    "shotRangeStop": 400000,
    "content": "geometry of the saddle coils for MAST",
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
    "creatorCode": "python create_netcdf_saddles.py",
    "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
    "createdBy": "sfrankel",
    "testCode": "",
    "testDate": "",
    "testedBy": "",
    }
        
    xmb_parquet_to_netcdf("geometry/saddle.nc", headerdict)