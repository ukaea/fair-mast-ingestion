import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import fnmatch

def create_header(ncfile, headerdict):
    for key, value in headerdict.items():
        setattr(ncfile, key, value)

def set_orientation(data, orientation):
    """Set unit vectors based on orientation. Note orientation has had spaces removed and is in uppercase."""
    if "INVERTED" not in orientation:
        if "VERTICAL" in orientation:
            data["orientation"]["measurement_direction"] = "VERTICAL"
            data["orientation"]["unit_vector"]["r"] = 0.
            data["orientation"]["unit_vector"]["phi"] = 0.
            data["orientation"]["unit_vector"]["z"] = 1.
        elif "RADIAL" in orientation:
            data["orientation"]["measurement_direction"] = "RADIAL"
            data["orientation"]["unit_vector"]["r"] = 1.
            data["orientation"]["unit_vector"]["phi"] = 0.
            data["orientation"]["unit_vector"]["z"] = 0.
        elif "TOROIDAL" in orientation:
            data["orientation"]["measurement_direction"] = "TOROIDAL"
            data["orientation"]["unit_vector"]["r"] = 0.
            data["orientation"]["unit_vector"]["phi"] = 1.
            data["orientation"]["unit_vector"]["z"] = 0.
    else:
        if "VERTICAL" in orientation:
            data["orientation"]["measurement_direction"] = "VERTICAL (INVERTED)"
            data["orientation"]["unit_vector"]["r"] = 0.
            data["orientation"]["unit_vector"]["phi"] = 0.
            data["orientation"]["unit_vector"]["z"] = -1.
        elif "RADIAL" in orientation:
            data["orientation"]["measurement_direction"] = "RADIAL (INVERTED)"
            data["orientation"]["unit_vector"]["r"] = -1.
            data["orientation"]["unit_vector"]["phi"] = 0.
            data["orientation"]["unit_vector"]["z"] = 0.
        elif "TOROIDAL" in orientation:
            data["orientation"]["measurement_direction"] = "TOROIDAL (INVERTED)"
            data["orientation"]["unit_vector"]["r"] = 0.
            data["orientation"]["unit_vector"]["phi"] = -1.
            data["orientation"]["unit_vector"]["z"] = 0.

def create_omaha_variable(group, parquet_file, var_type, version):
    """Create omaha variables and add to group. Use omaha parquet files."""
    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))

        data = np.empty(1, var_type.dtype_view)
        data["name"][:] = row["uda_name"].replace("/", "_")
        data["version"] = version
        data["coordinate"]["r"] = row["r"]
        data["coordinate"]["z"] = row["z"]
        data["coordinate"]["phi"] = row["toroidal_angle"]
        # data["geometry"]["length"] = row["length"] # this information is not provided
        data["geometry"]["nturnsLayer1"] = 28
        data["geometry"]["nturnsLayer2"] = 28
        data["geometry"]["nturnsTotal"] = 56
        data["geometry"]["areaLayer1"] = 0.037
        data["geometry"]["areaLayer2"] = 0.037
        data["geometry"]["areaAve"] = 0.037 / 28
        cleaned_orientation = (str(row["orientation"]).replace(" ", "")).upper()
        set_orientation(data, cleaned_orientation)
        var[:] = data
        #var.setnccattr("units", "SI units: degrees, m")

def omaha_parquet_to_netcdf(netcdf_file, headerdict):
    """Convert parquet file to netcdf."""
    with Dataset(netcdf_file, "w", format="NETCDF4") as ncfile:

        """Add global attributes."""
        create_header(ncfile, headerdict)

        """Create omaha coils group."""
        omaha_group = ncfile.createGroup("omahagroup")

        """Not sure how to group the omaha coils."""
        pre_22108 = omaha_group.createGroup("pre_22108")
        from22108to23945 = omaha_group.createGroup("from22108to23945")
        post_23945 = omaha_group.createGroup("post_23945")

        """Define data types and combine into compound data types."""
        coord_dtype = np.dtype([("r", "<f8"), ("z", "<f8"), ("phi", "<f8")])
        unitvector_dtype = np.dtype([("r", ">i4"), ("z", ">i4"), ("phi", ">i4")])
        geometry_dtype = np.dtype([
            ("nturnsLayer1", "<f8"),
            ("nturnsLayer2", "<f8"),
            ("nturnsTotal", "<f8"),
            ("areaLayer1", "<f8"),
            ("areaLayer2", "<f8"),
            ("areaAve", "<f8")
        ])
        orientation_dtype = np.dtype([
            ("measurement_direction", "S30"),
            ("unit_vector", unitvector_dtype)
        ])
        omaha_dtype = np.dtype([
            ("name", "S10"),
            ("version", "<f8"),
            ("orientation", orientation_dtype),
            ("coordinate", coord_dtype),
            ("geometry", geometry_dtype)
        ])

        omaha_group.createCompoundType(unitvector_dtype, "UNITVECTORS")
        omaha_group.createCompoundType(orientation_dtype, "ORIENTATION")
        omaha_group.createCompoundType(coord_dtype, "COORDINATES")
        omaha_group.createCompoundType(geometry_dtype, "GEOMETRY")
        
        var_type = omaha_group.createCompoundType(omaha_dtype, "OMAHA")

        omaha_group.createDimension("singleDim", 1)

        """Assign parquet file contents to appropriate groups."""
        parquet_files = {
            "pre_22108": ("geometry/data/omaha_files/omaha_pre_22108.parquet", "PRE 22108"),
            "xmc_post_23945": ("geometry/data/omaha_files/xmc_omaha_post_23945.parquet", "XMC POST 23945"),
            "xmd_22108_23945": ("geometry/data/omaha_files/xmd_omaha_22108_23945.parquet", "XMD BETWEEN"),
            "xmd_post_23945": ("geometry/data/omaha_files/xmd_omaha_post_23945.parquet", "XMD POST"),
            "xmo_22108_23945": ("geometry/data/omaha_files/xmo_omaha_22108_23945.parquet", "XMO BETWEEN")
        }

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        """Note: the patterns are for the parquet_files KEYS above."""
        patterns = ["pre*", "*post*", "*22108_23945"]
        groups = [pre_22108, post_23945, from22108to23945]

        keys = list(parquet_files.keys())
        
        for i in range(0,len(patterns)): # indexing through patterns
            
            matching_keys = fnmatch.filter(keys, patterns[i]) # outputs list of pq file keys that match a pattern
            matching_values = list(dict((k, parquet_files[k]) for k in matching_keys).values()) # gets filepaths corresponding to keys that match the pattern
            matching_filepaths = []

            for q in matching_values:
                matching_filepaths.append(q[0])
            
            # print(matching_filepaths)
            for n in matching_filepaths:
                create_omaha_variable(groups[i], n, var_type, version) # add variable to matching group



if __name__ == "__main__":
    # Metadata for the NetCDF file
    headerdict = {
    "Conventions": "",
    "device": "MAST",
    "class": "magnetics",
    "system": "omaha",
    "configuration": "geometry",
    "shotRangeStart": 0,
    "shotRangeStop": 40000,
    "content": "geometry of the omaha coils for MAST",
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
    "creatorCode": "python create_netcdf_omaha.py",
    "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
    "createdBy": "sfrankel",
    "testCode": "",
    "testDate": "",
    "testedBy": "",
    }
        
    omaha_parquet_to_netcdf("geometry/omaha.nc", headerdict)