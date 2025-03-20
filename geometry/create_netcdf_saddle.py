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
        #data["sector"] = row["sector"]
        data["centre"]["r"] = row["r"]
        data["centre"]["z"] = row["z"]
        data["centre"]["phi"] = row["toroidal_angle"]
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

        magnetics = ncfile.createGroup("magnetics")

        """Create saddle coils group."""
        saddlecoils = magnetics.createGroup("saddlecoils")
        
        # no centrecol/outervessel/psp info available

        """Group by lower, middle and upper saddle coils."""
        upper_group = saddlecoils.createGroup("upper")
        middle_group = saddlecoils.createGroup("middle")
        lower_group = saddlecoils.createGroup("lower")
        
        

        """Define data types and combine into compound data types."""
        coord_dtype = np.dtype([("r", "<f8"), ("z", "<f8"), ("phi", "<f8")])
        path_dtype = np.dtype([("r", "<f8"), ("z", "<f8"), ("phi", "<f8")])
        geometry_dtype = np.dtype([
            ("height", "<f8"),
            ("width", "<f8"),
            ("cornerRadius", "<f8"),
            ("area", "<f8"),
            ("angle", "<f8")
        ])

        saddle_dtype = np.dtype([
            ("name", "S10"),
            ("description", "S30"),
            ("location", "S30"),
            ("refFrame", "S10"),
            ("status", "S10"),
            ("version", "<f8"),
            ("saddleType", "S10"),
            ("centre", coord_dtype),
            ("coilPath", path_dtype),
            ("geometry", geometry_dtype)
        ])

        saddlecoils.createCompoundType(coord_dtype, "COORDINATE")
        saddlecoils.createCompoundType(path_dtype, "COORDINATE_PATH")
        saddlecoils.createCompoundType(geometry_dtype, "GEOMETRY")
        
        var_type = saddlecoils.createCompoundType(saddle_dtype, "SADDLECOIL")

        saddlecoils.createDimension("singleDim", 1)

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
    "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
    "coordinateSystem": "cylindrical",
    "device": "MAST",
    "shotRangeStart": 0,
    "shotRangeStop": 400000,
    "createdBy": "sfrankel",
    "system": "saddlecoils",
    "signedOffDate": "",
    "class": "magnetics",
    "units": "SI, degrees, m",
    "version": 0,
    "revision": 0,
    "conventions": "MAST MetaData",
    "status": "development",
    "releaseDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
    "releaseTime": datetime.strftime(datetime.now(), "%H:%M:%S"),
    "creatorCode": "python create_netcdf_saddles.py",
    "owner": "jhodson",
    "signedOffBy": "",
    "configuration": "geometry",
    "content": "geometry of the saddle coils for MAST",
    "comment": "",
    "structureCastType": "unknown",
    "calibration": "None",
    "testCode": "",
    "testDate": "",
    "testedBy": "",
    }
        
    xmb_parquet_to_netcdf("geometry/saddle.nc", headerdict)