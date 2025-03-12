import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset
import fnmatch

def create_header(ncfile, headerdict):
    for key, value in headerdict.items():
        setattr(ncfile, key, value)

# information not provided
def set_orientation(data, poloidal_angle):
    """Set orientation based on poloidal angle."""
    if poloidal_angle == 90:
        data["orientation"]["measurement_direction"] = "PARALLEL"
        data["orientation"]["unit_vector"]["r"] = 0.
        data["orientation"]["unit_vector"]["phi"] = 0.
        data["orientation"]["unit_vector"]["z"] = 1.
    elif poloidal_angle == 0:
        data["orientation"]["measurement_direction"] = "NORMAL"
        data["orientation"]["unit_vector"]["r"] = 1.
        data["orientation"]["unit_vector"]["phi"] = 0.
        data["orientation"]["unit_vector"]["z"] = 0.
    else:
        data["orientation"]["measurement_direction"] = "TOROIDAL"
        data["orientation"]["unit_vector"]["r"] = 0.
        data["orientation"]["unit_vector"]["phi"] = 1.
        data["orientation"]["unit_vector"]["z"] = 0.

def create_mirnov_variable(group, parquet_file, var_type, version):
    """Create mirnov coil variables and add to group. Use XMC parquet files."""
    df = pd.read_parquet(parquet_file)
    used_names = []
    for _, row in df.iterrows():

        namevar = str(row["uda_name"].replace("/", "_"))
        data = np.empty(1, var_type.dtype_view)

        if namevar not in used_names: # make sure no duplicates make it through
            var = group.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))
            data["name"][:] = row["uda_name"].replace("/", "_")
            used_names.append(namevar)
        else:
            continue
        data["version"] = version
        if "r" in df.columns:
            data["coordinate"]["r"] = row["r"]
        else:
            continue
        if "z" in df.columns:
            data["coordinate"]["z"] = row["z"]
        elif "z1" in df.columns():
            data["coordinate"]["z1"] = row["z1"]
            data["coordinate"]["z2"] = row["z2"]
            data["coordinate"]["z3"] = row["z3"]
        data["coordinate"]["phi"] = row["toroidal_angle"]
        # data["geometry"]["length"] = row["length"] # this information is not provided
        
        # set these values correctly from pdfs
        data["geometry"]["nturnsLayer1"] = 28
        data["geometry"]["nturnsLayer2"] = 28
        data["geometry"]["nturnsTotal"] = 56
        data["geometry"]["areaLayer1"] = 0.037
        data["geometry"]["areaLayer2"] = 0.037
        data["geometry"]["areaAve"] = 0.037 / 28

        # set_orientation(data, row["poloidal_angle"]) # issue: no poloidal angle provided

        var[:] = data
        #var.setnccattr("units", "SI units: degrees, m")

def xmc_parquet_to_netcdf(netcdf_file, headerdict):
    """Convert parquet file to netcdf."""
    with Dataset(netcdf_file, "w", format="NETCDF4") as ncfile:

        """Add global attributes."""
        create_header(ncfile, headerdict)

        """Create mirnov coils group."""
        magnetics = ncfile.createGroup("magnetics")
        mirnov_group = magnetics.createGroup("mirnov")

        """Group by central column (vertical and toroidal) and outer vessel wall (vertical) Mirnov arrays."""
        centralcolumn_group = mirnov_group.createGroup("centrecolumn")
        vessel_group = mirnov_group.createGroup("outervessel")

        """Define data types and combine into compound data types."""
        unitvector_dtype = np.dtype([("length", "S30"), ("areaAve", "S30")])
        coord_dtype = np.dtype([("r", "<f8"), ("z", "<f8"), ("phi", "<f8")])
        geometry_dtype = np.dtype([
            ("length", "<f8"),
            ("nturnsLayer1", "<f8"),
            ("nturnsLayer2", "<f8"),
            ("nturnsTotal", "<f8"),
            ("areaLayer1", "<f8"),
            ("areaLayer2", "<f8"),
            ("areaAve", "<f8")
        ])
        orientation_dtype = np.dtype([
            ("unit_vector", coord_dtype),
            ("measurement_direction", "S30")
        ])

        mirnov_dtype = np.dtype([
            ("name", "S30"),
            ("version", "<f8"),
            ("orientation", orientation_dtype),
            ("coordinate", coord_dtype),
            ("geometry", geometry_dtype)
        ])

        mirnov_group.createCompoundType(coord_dtype, "COORDINATE")
        mirnov_group.createCompoundType(geometry_dtype, "GEOMETRY")
        mirnov_group.createCompoundType(unitvector_dtype, "GEOMETRYUNITS")
        mirnov_group.createCompoundType(orientation_dtype, "ORIENTATION")
        
        
        
        var_type = mirnov_group.createCompoundType(mirnov_dtype, "MIRNOV")

        mirnov_group.createDimension("singleDim", 1)

        """Assign parquet file contents to appropriate groups."""
        parquet_files = {
            "centralcolumn_toroidal": ("geometry/data/xmc/ccmt.parquet", "CENTRALCOL TOROIDAL"),
            "centralcolumn_vertical": ("geometry/data/xmc/ccmv.parquet", "CENTRALCOL VERTICAL"),
            "outboard_vertical": ("geometry/data/xmc/xmc_omv.parquet", "OUTBOARD (VESSEL WALL) VERTICAL")
        }

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        patterns = ["centralcolumn*", "outboard*"]
        groups = [centralcolumn_group, vessel_group]

        keys = list(parquet_files.keys())
        
        for i in range(0,len(patterns)): # indexing through patterns
            
            matching_keys = fnmatch.filter(keys, patterns[i]) # outputs list of pq file keys that match a pattern
            matching_values = list(dict((k, parquet_files[k]) for k in matching_keys).values()) # gets filepaths corresponding to keys that match the pattern
            matching_filepaths = []

            for q in matching_values:
                matching_filepaths.append(q[0])
            
            # print(matching_filepaths)
            for n in matching_filepaths:
                create_mirnov_variable(groups[i], n, var_type, version) # add variable to matching group



if __name__ == "__main__":
    # Metadata for the NetCDF file
    headerdict = {
    "Conventions": "",
    "device": "MAST",
    "class": "magnetics",
    "system": "mirnovcoils",
    "configuration": "geometry",
    "shotRangeStart": 0,
    "shotRangeStop": 400000,
    "content": "geometry of the mirnov coils for MAST",
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
    "creatorCode": "python create_netcdf_mirnovs.py",
    "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
    "createdBy": "sfrankel",
    "testCode": "",
    "testDate": "",
    "testedBy": "",
    }
        
    xmc_parquet_to_netcdf("geometry/mirnovs.nc", headerdict)
