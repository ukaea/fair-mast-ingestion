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
