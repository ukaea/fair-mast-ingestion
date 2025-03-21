import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

def create_header(ncfile, headerdict):
    for key, value in headerdict.items():
        setattr(ncfile, key, value)

def create_passive_variable_mid(group, parquet_file, var_type, version):
    df = pd.read_parquet(parquet_file)
    df_ou, df_ol, df_iu, df_il = df[:3], df[3:6], df[6:9], df[9:12]
    
    subgroups = ["mid_ou", "mid_ol", "mid_iu", "mid_il"]
    dfs = [df_ou, df_ol, df_iu, df_il]

    for i in range(len(subgroups)):
        subgroup = group.createGroup(subgroups[i])
        df = dfs[i]
        for _, row in df.iterrows():
            var = subgroup.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))

            data = np.empty(1, var_type.dtype_view)
            data["name"][:] = row["uda_name"].replace("/", "_")
            data["version"] = version
            data["phi_cut"] = 0
            data["centreR"] = row["r"]
            data["centreZ"] = row["z"]
            data["dR"] = row["dR"]
            data["dZ"] = row["dZ"]
            data["shapeAngle1"] = row["ang1"] if row["ang1"] >= 0 else np.float64(row["ang1"] + 360.0)
            data["shapeAngle2"] = row["ang2"] if row["ang2"] >= 0 else np.float64(row["ang2"] + 360.0)
            data["resistivity"] = row["resistivity"]
            data["resistivityUnits"] = "mOhms"

            var[:] = data

def create_passive_variable(group, parquet_file, var_type, version):
    """Create passive structure variables and add to group. Use AMM parquet files."""
    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))

        data = np.empty(1, var_type.dtype_view)
        data["name"][:] = row["uda_name"].replace("/", "_")
        data["version"] = version
        data["phi_cut"] = 0
        data["centreR"] = row["r"]
        data["centreZ"] = row["z"]
        data["dR"] = row["dR"]
        data["dZ"] = row["dZ"]
        data["shapeAngle1"] = row["ang1"] if row["ang1"] >= 0 else row["ang1"] + 360
        data["shapeAngle2"] = row["ang2"] if row["ang2"] >= 0 else row["ang2"] + 360
        data["resistivity"] = row["resistivity"] 
        data["resistivityUnits"] = "mOhms" #looks like mili-Ohm, but looking at the MAST-U data it might be resistivity i.e. Ohm m

        var[:] = data

def amm_parquet_to_netcdf(netcdf_file, headerdict):
    """Convert parquet file to netcdf."""
    with Dataset(netcdf_file, "w", format="NETCDF4") as ncfile:

        """Add global attributes."""
        create_header(ncfile, headerdict)

        passive_dtype = np.dtype([
            ("name", "S50"),
            ("version", "S50"),
            ("phi_cut", "f4"),
            ("centreR", "f4"),
            ("centreZ", "f4"),
            ("dR", "f4"),
            ("dZ", "f4"),
            ("shapeAngle1", "f4"),
            ("shapeAngle2", "f4"),
            ("resistivity", "f4"),
            ("resistivityUnits", "S50")
        ])

        """Create passive structures and efit group."""
        passive_group = ncfile.createGroup("passive")
        efit_group = passive_group.createGroup("efit")

        var_type = passive_group.createCompoundType(passive_dtype, "PASSIVE")
        passive_group.createDimension("singleDim", 1)

        """Create branches of passive structures group: central column, walls (horizontal & other), and P2 (lower & upper)."""
        centralcolumn_group = efit_group.createGroup("centralcolumn") # topcol and botcol, but may be short for colusseum instead. may also contain incon?
        wall_group = efit_group.createGroup("walls") # upper & lower horizontal, and vertical
        p2_group = efit_group.createGroup("p2")

        passive_subgroups = {
            "incon" : centralcolumn_group.createGroup("incon"),
            "ring_rodgr": centralcolumn_group.createGroup("ring_rodgr"),
            "uhorw": wall_group.createGroup("uhorw"),
            "lhorw": wall_group.createGroup("lhorw"),
            "vertw": wall_group.createGroup("vertw"),
            "p2larm": p2_group.createGroup("p2larm"),
            "p2uarm": p2_group.createGroup("p2uarm"),
            "p2ldivpl": p2_group.createGroup("p2ldivpl"),
            "p2udivpl": p2_group.createGroup("p2udivpl")   
        }

        parquet_files = {
            "topcol": ["geometry/data/amm/amm_topcol.parquet"],
            "botcol": ["geometry/data/amm/amm_botcol.parquet"],
            "endcrown_l": ["geometry/data/amm/amm_endcrown_l.parquet"],
            "endcrown_u": ["geometry/data/amm/amm_endcrown_u.parquet"],
            "incon": ["geometry/data/amm/amm_incon.parquet"],
            "ring_rodgr": ["geometry/data/amm/amm_rodgr.parquet", "geometry/data/amm/amm_ring.parquet"],
            "uhorw": ["geometry/data/amm/amm_uhorw.parquet"],
            "lhorw": ["geometry/data/amm/amm_lhorw.parquet"],
            "vertw": ["geometry/data/amm/amm_vertw.parquet"],
            "mid": ["geometry/data/amm/amm_mid.parquet"],
            "p2larm": ["geometry/data/amm/amm_p2larm1.parquet", "geometry/data/amm/amm_p2larm2.parquet", "geometry/data/amm/amm_p2larm3.parquet"],
            "p2uarm": ["geometry/data/amm/amm_p2uarm1.parquet", "geometry/data/amm/amm_p2uarm2.parquet", "geometry/data/amm/amm_p2uarm3.parquet"],
            "p2ldivpl": ["geometry/data/amm/amm_p2ldivpl1.parquet", "geometry/data/amm/amm_p2ldivpl2.parquet"],
            "p2udivpl": ["geometry/data/amm/amm_p2udivpl1.parquet", "geometry/data/amm/amm_p2udivpl2.parquet"]
        }

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        cc = ["topcol", "botcol", "endcrown_l", "endcrown_u"]
        for subgroup_key, (file_path) in parquet_files.items():

            if subgroup_key in cc:
                create_passive_variable(centralcolumn_group, file_path, var_type, version)
            elif subgroup_key == "mid":
                create_passive_variable_mid(wall_group, file_path, var_type, version)
            else:
                create_passive_variable(passive_subgroups[subgroup_key], file_path, var_type, version)

if __name__ == "__main__":
    # Metadata for the NetCDF file
    headerdict = {
        "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
        "coordinateSystem": "cylindrical",
        "device": "MAST",
        "shotRangeStart": "0LL",
        "shotRangeStop": "40000LL",
        "createdBy": "sfrankel",
        "system": "passive structure",
        "signal": "passive structure",
        "signedOffDate": "",
        "class": "passive structure",
        "units": "SI, degrees",
        "version": 1,
        "revision": 0,
        "conventions": "",
        "status": "development",
        "releaseDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
        "releaseTime": datetime.strftime(datetime.now(), "%H:%M:%S"),
        "creatorCode": "python create_netcdf_passive.py",
        "owner": "sfrankel",
        "signedOffBy": "ldormangajic",
        } 
    amm_parquet_to_netcdf("geometry/passivestructures.nc", headerdict)