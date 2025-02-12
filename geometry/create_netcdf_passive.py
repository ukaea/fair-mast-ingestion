import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

def add_variables_to_group(group, parquet_file, var_type, location, version):
    """Create passive structure variables and add to group. Use AMM parquet files."""
    df = pd.read_parquet(parquet_file)

    for _, row in df.iterrows():
        var = group.createVariable(row["uda_name"].replace("/", "_"), var_type, ("singleDim",))

    data = np.empty(1, var_type.dtype_view)
    data["name"][:] = row["uda_name"].replace("/", "_")
    data["version"] = version
    data["location"] = location
    data["circuit_number"] = row["circuit_number"]
    data["coordinate"]["r"] = row["r"]
    data["coordinate"]["z"] = row["z"]
    data["dimensions"]["height"] = row["height"]
    data["dimensions"]["width"] = row["width"]
    data["angle"]["ang1"] = row["ang1"]
    data["angle"]["ang2"] = row["ang2"]

    var[:] = data
    var.setnccattr("units", "SI units: degrees, m")

def amm_parquet_to_pdf(netcdf_file, headerdict):
    """Convert parquet file to netcdf."""
    with Dataset(netcdf_file, "w", format="NETCDF4") as ncfile:

        # add global attributes

        for key, value in headerdict.items():
            setattr(ncfile, key, value)

        # create passive structures group

        passive_group = ncfile.createGroup("passivegroup")
        centralcolumn_group = passive_group.createGroup("centralcolumn") # topcol and botcol, but may be short for colusseum instead. may also contain incon?
        wall_group = passive_group.createGroup("walls") # upper & lower horizontal, and vertical
        wall_group_horizontal = wall_group.createGroup("horizontal")
        p2_group = passive_group.createGroup("p2")
        # mid divides cross section diagram into 3
        # incon: maybe in the column?
        # col may be colusseum

        passive_subgroups = {
            "centralcolumn_upper": centralcolumn_group.createGroup("upper"),
            "centralcolumn_lower": centralcolumn_group.createGroup("lower"),
            "wall_group_vertical": wall_group.createGroup("vertical"),
            "p2_lower": p2_group.createGroup("lower"),
            "p2_upper": p2_group.createGroup("upper"),
            
        }

        passive_subsubgroups = {
            "wall_horiz_upper": wall_group_horizontal.createGroup("upper"),
            "wall_horiz_lower": wall_group_horizontal.createGroup("lower")
        }



        

        






