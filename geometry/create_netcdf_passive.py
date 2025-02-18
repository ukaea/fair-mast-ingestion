import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

def create_header(ncfile, headerdict):
    for key, value in headerdict.items():
        setattr(ncfile, key, value)

def create_passive_variable(group, parquet_file, var_type, version):
    """Create passive structure variables and add to group. Use AMM parquet files."""
    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))

        data = np.empty(1, var_type.dtype_view)
        data["name"][:] = row["uda_name"].replace("/", "_")
        data["version"] = version
        #data["location"] = location
        data["circuit_number"] = row["circuit_number"]
        data["coordinate"]["centreR"] = row["r"]
        data["coordinate"]["centreZ"] = row["z"]
        data["dimensions"]["dR"] = row["width"]
        data["dimensions"]["dZ"] = row["height"]
        data["angle"]["shapeAngle1"] = row["ang1"]
        data["angle"]["shapeAngle2"] = row["ang2"]

        var[:] = data
        #var.setnccattr("units", "SI units: degrees, m")

def amm_parquet_to_netcdf(netcdf_file, headerdict):
    """Convert parquet file to netcdf."""
    with Dataset(netcdf_file, "w", format="NETCDF4") as ncfile:

        """Add global attributes."""
        create_header(ncfile, headerdict)

        """Create passive structures group."""
        passive_group = ncfile.createGroup("passivegroup")

        """Create branches of passive structures group: central column, walls (horizontal & other), endcrown and P2 (lower & upper)."""
        centralcolumn_group = passive_group.createGroup("centralcolumn") # topcol and botcol, but may be short for colusseum instead. may also contain incon?
        wall_group = passive_group.createGroup("walls") # upper & lower horizontal, and vertical
        wall_group_horizontal = wall_group.createGroup("horizontal")
        endcrown_group = passive_group.createGroup("endcrown")
        p2_group = passive_group.createGroup("p2")
        p2_lower = p2_group.createGroup("lower")
        p2_upper = p2_group.createGroup("upper")
        
        """Create subgroups of the above groups and branches."""
        passive_subgroups = {
            "centralcolumn_upper": centralcolumn_group.createGroup("upper"),
            "centralcolumn_lower": centralcolumn_group.createGroup("lower"),
            "p2_larm1": p2_lower.createGroup("arm1"),
            "p2_larm2": p2_lower.createGroup("arm2"),
            "p2_larm3": p2_lower.createGroup("arm3"),
            "p2_ldivpl1": p2_lower.createGroup("divplate1"), # divertor plate
            "p2_ldivpl2": p2_lower.createGroup("divplate2"),
            "p2_uarm1": p2_upper.createGroup("arm1"),
            "p2_uarm2": p2_upper.createGroup("arm2"),
            "p2_uarm3": p2_upper.createGroup("arm3"),
            "p2_udivpl1": p2_upper.createGroup("divplate1"),
            "p2_udivpl2": p2_upper.createGroup("divplate2"),
            "wall_horiz_upper": wall_group_horizontal.createGroup("upper"),
            "wall_horiz_lower": wall_group_horizontal.createGroup("lower"),
            "wall_group_vertical": wall_group.createGroup("vertical"),
            "mid": passive_group.createGroup("mid"),
            "endcrown_lower": endcrown_group.createGroup("lower"),
            "endcrown_upper": endcrown_group.createGroup("upper"),
            "incon": centralcolumn_group.createGroup("incon"),
            "rodgr": centralcolumn_group.createGroup("rodgr"),
            "ring": passive_group.createGroup("ring")
        }

        coord_dtype = np.dtype([("centreR", "<f8"), ("centreZ", "<f8")])

        dims_dtype = np.dtype([("dR", "<f8"), ("dZ", "<f8")])
        
        geometry_dtype = np.dtype([
            #("phi_cut", "<f8"),
            ("shapeAngle1", "<f8"),
            ("shapeAngle2", "<f8")
        ])

        passive_dtype = np.dtype([
            ("name", "S10"),
            ("version", "<f8"),
            ("circuit_number", "<f8"),
            ("coordinate", coord_dtype),
            ("dimensions", dims_dtype),
            ("angle", geometry_dtype)
            #("material", "b"),
            #("elementLabels", "8b"), # array of less than 8 S50s
            #("efitGroup", "b"),
            #("resistivity", "<f8"),
            #("resistivityError", "<f8"),
            #("resistivityUnits", "b"), # resistivity only in PDFs not Parquets
        ])


        passive_group.createCompoundType(geometry_dtype, "GEOMETRY")
        passive_group.createCompoundType(dims_dtype, "DIMENSIONS")
        passive_group.createCompoundType(coord_dtype, "COORDINATES")
        var_type = passive_group.createCompoundType(passive_dtype, "PASSIVE")

        passive_group.createDimension("singleDim", 1)

        parquet_files = {
            "centralcolumn_upper": ("geometry/data/amm/amm_topcol.parquet", "CENTRALCOL UPPER"),
            "centralcolumn_lower": ("geometry/data/amm/amm_botcol.parquet", "CENTRALCOL LOWER"),
            "p2_larm1": ("geometry/data/amm/amm_p2larm1.parquet", "LOWER ARM 1"),
            "p2_larm2": ("geometry/data/amm/amm_p2larm2.parquet", "LOWER ARM 2"),
            "p2_larm3": ("geometry/data/amm/amm_p2larm3.parquet", "LOWER ARM 3"),
            "p2_ldivpl1": ("geometry/data/amm/amm_p2ldivpl1.parquet", "LOWER DIVERTOR PLATE 1"), # divertor plate
            "p2_ldivpl2": ("geometry/data/amm/amm_p2ldivpl2.parquet", "LOWER DIVERTOR PLATE 2"),
            "p2_uarm1": ("geometry/data/amm/amm_p2uarm1.parquet", "UPPER ARM 1"),
            "p2_uarm2": ("geometry/data/amm/amm_p2uarm2.parquet", "UPPER ARM 2"),
            "p2_uarm3": ("geometry/data/amm/amm_p2uarm3.parquet", "UPPER ARM 3"),
            "p2_udivpl1": ("geometry/data/amm/amm_p2udivpl1.parquet", "UPPER DIVERTOR PLATE 1"),
            "p2_udivpl2": ("geometry/data/amm/amm_p2udivpl2.parquet", "UPPER DIVERTOR PLATE 2"),
            "wall_horiz_upper": ("geometry/data/amm/amm_uhorw.parquet", "UPPER HORIZONTAL WALL"),
            "wall_horiz_lower": ("geometry/data/amm/amm_lhorw.parquet", "LOWER HORIZONTAL WALL"),
            "wall_group_vertical": ("geometry/data/amm/amm_vertw.parquet", "VERTICAL WALL"),
            "mid": ("geometry/data/amm/amm_mid.parquet", "MID"),
            "endcrown_lower": ("geometry/data/amm/amm_endcrown_l.parquet", "LOWER ENDCROWN"),
            "endcrown_upper": ("geometry/data/amm/amm_endcrown_u.parquet", "UPPER ENDCROWN"),
            "incon": ("geometry/data/amm/amm_incon.parquet", "INCON"),
            "rodgr": ("geometry/data/amm/amm_rodr.parquet", "RODGR"), # not a typo
            "ring": ("geometry/data/amm/amm_ring.parquet", "RING")

        }

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        for subgroup_key, (file_path) in parquet_files.items():
            create_passive_variable(
                passive_subgroups[subgroup_key], file_path[0], var_type, version
            )

if __name__ == "__main__":
    # Metadata for the NetCDF file
    headerdict = {
    "Conventions": "",
    "device": "MAST",
    "class": "magnetics",
    "system": "passivestructures",
    "configuration": "geometry",
    "shotRangeStart": 0,
    "shotRangeStop": 400000,
    "content": "geometry of the passive structures for MAST",
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
    "creatorCode": "python create_netcdf_passive.py",
    "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
    "createdBy": "sfrankel",
    "testCode": "",
    "testDate": "",
    "testedBy": "",
    }
        
    amm_parquet_to_netcdf("geometry/passivestructures.nc", headerdict)


        

        






