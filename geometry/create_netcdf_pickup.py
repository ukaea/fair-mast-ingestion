import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

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

def add_variables_to_group(df, group, var_type, version, toroidal_angle_col):
    """Add variables to a NetCDF group from a DataFrame."""
    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"], var_type, ("singleDim",))
        data = np.empty(1, var_type.dtype_view)
        
        # Populate the data structure
        data["name"][:] = row["uda_name"]
        data["version"] = version
        data["coordinate"]["r"] = row["r"]
        data["coordinate"]["z"] = row["z"]
        data["coordinate"]["phi"] = row[toroidal_angle_col]
        data["geometry"]["length"] = row["length"]

        set_orientation(data, row["poloidal_angle"])
        var[:] = data
        var.setncattr("units", "SI units: degrees, m")

def add_2layer_variables_to_group(df, group, var_type, version, toroidal_angle_col):
    """Add variables with 2-layer geometry to a NetCDF group."""
    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"], var_type, ("singleDim",))
        data = np.empty(1, var_type.dtype_view)
        
        # Populate the data structure
        data["name"][:] = row["uda_name"]
        data["version"] = version
        data["coordinate"]["r"] = row["r"]
        data["coordinate"]["z"] = row["z"]
        data["coordinate"]["phi"] = row[toroidal_angle_col]
        data["geometry"]["length"] = row["length"]
        data["geometry"]["nturnsLayer1"] = 28
        data["geometry"]["nturnsLayer2"] = 28
        data["geometry"]["nturnsTotal"] = 56
        data["geometry"]["areaLayer1"] = 0.037
        data["geometry"]["areaLayer2"] = 0.037
        data["geometry"]["areaAve"] = 0.037 / 28

        set_orientation(data, row["poloidal_angle"])
        var[:] = data
        var.setncattr("units", "SI units: degrees, m, m^2")

def parquet_to_netcdf(netcdf_file, headerdict):
    """Convert Parquet files to a NetCDF file."""
    with Dataset(netcdf_file, 'w', format='NETCDF4') as ncfile:
        # Add global attributes
        for key, value in headerdict.items():
            setattr(ncfile, key, value)

        # Create groups
        mag_grp = ncfile.createGroup("magnetics")
        pickup_group = mag_grp.createGroup("pickup")
        centrecol_group = pickup_group.createGroup("centrecolumn")
        cc_t1_group = centrecol_group.createGroup("t1")
        cc_t2_group = centrecol_group.createGroup("t2")

        outer_vessel_group = pickup_group.createGroup("outervessel")
        ov_t1_group = outer_vessel_group.createGroup("t1")
        ov_t2_group = outer_vessel_group.createGroup("t2")

        # Define data types
        unitvector_dtype = np.dtype([("r", ">i4"), ("z", ">i4"), ("phi", ">i4")])
        coord_dtype = np.dtype([("r", "<f8"), ("z", "<f8"), ("phi", "<f8")])
        geom_dtype = np.dtype([("length", "<f8")])
        geom_2layers_dtype = np.dtype([
            ("length", "<f8"),
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
        bv_dtype = np.dtype([
            ("name", "S50"),
            ("version", "<f8"),
            ("orientation", orientation_dtype),
            ("coordinate", coord_dtype),
            ("geometry", geom_dtype)
        ])
        bv_2layers_dtype = np.dtype([
            ("name", "S50"),
            ("version", "<f8"),
            ("orientation", orientation_dtype),
            ("coordinate", coord_dtype),
            ("geometry", geom_2layers_dtype)
        ])

        # Create compound types
        pickup_group.createCompoundType(unitvector_dtype, "UNIT_VECTOR")
        pickup_group.createCompoundType(orientation_dtype, "ORIENTATION")
        pickup_group.createCompoundType(coord_dtype, "COORDINATE")
        pickup_group.createCompoundType(geom_dtype, "GEOMETRY")
        pickup_group.createCompoundType(geom_2layers_dtype, "GEOMETRY_2LAYERS")
        lp_cp = pickup_group.createCompoundType(bv_dtype, "PICKUPCOIL")
        lp_2layers_cp = pickup_group.createCompoundType(bv_2layers_dtype, "PICKUPCOIL_2LAYERS")

        # Dimension
        pickup_group.createDimension("singleDim", 1)

        # Calculate version
        version = headerdict["version"] + 0.1 * headerdict["revision"]

        # Process Parquet files
        cc_df = pd.read_parquet("geometry/data/amb/ccbv.parquet")
        add_variables_to_group(cc_df, cc_t1_group, lp_cp, version, "toroidal_angle1")
        add_variables_to_group(cc_df, cc_t2_group, lp_cp, version, "toroidal_angle2")

        obr_df = pd.read_parquet("geometry/data/amb/xma_obr.parquet")
        add_2layer_variables_to_group(obr_df, ov_t1_group, lp_2layers_cp, version, "toroidal_angle1")
        add_2layer_variables_to_group(obr_df, ov_t2_group, lp_2layers_cp, version, "toroidal_angle2")

        obv_df = pd.read_parquet("geometry/data/amb/xma_obv.parquet")
        add_2layer_variables_to_group(obv_df, ov_t1_group, lp_2layers_cp, version, "toroidal_angle1")
        add_2layer_variables_to_group(obv_df, ov_t2_group, lp_2layers_cp, version, "toroidal_angle2")

if __name__ == "__main__":
    # Metadata for the NetCDF file
    headerdict = {
        "Conventions": "",
        "device": "MAST",
        "class": "magnetics",
        "system": "pickup",
        "configuration": "geometry",
        "shotRangeStart": 0,
        "shotRangeStop": 400000,
        "content": "geometry of the magnetic pickup coils for MAST",
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
        "creatorCode": "python create_netcdf_pickup.py",
        "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
        "createdBy": "jhodson",
        "testCode": "",
        "testDate": "",
        "testedBy": ""
    }

    # Create NetCDF file
    parquet_to_netcdf("geometry/pickup_coils.nc", headerdict)