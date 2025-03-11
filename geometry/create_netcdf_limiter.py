import pandas as pd
import numpy as np
from datetime import datetime
from netCDF4 import Dataset

# Define R and Z values
data = np.array([
    [1.9000000,  0.4050000],
    [1.5551043,  0.4050000],
    [1.5551043,  0.8225002],
    [1.4079306,  0.8225002],
    [1.4079306,  1.0330003],
    [1.0399311,  1.0330003],
    [1.0399311,  1.1950001],
    [1.9000000,  1.1950001],
    [1.9000000,  1.8250000],
    [0.5649307,  1.8250000],
    [0.5649307,  1.7280816],
    [0.7835000,  1.7280816],
    [0.7835000,  1.7155817],
    [0.5825903,  1.5470001],
    [0.4165000,  1.5470001],
    [0.2800000,  1.6835001],
    [0.2800000,  1.2290885],
    [0.1952444,  1.0835000],
    [0.1952444, -1.0835000],
    [0.2800000, -1.2290885],
    [0.2800000, -1.6835001],
    [0.4165000, -1.5470001],
    [0.5825903, -1.5470001],
    [0.7835000, -1.7155817],
    [0.7835000, -1.7280816],
    [0.5649307, -1.7280816],
    [0.5649307, -1.8250000],
    [1.9000000, -1.8250000],
    [1.9000000, -1.1950001],
    [1.0399311, -1.1950001],
    [1.0399311, -1.0330003],
    [1.4079306, -1.0330003],
    [1.4079306, -0.8225002],
    [1.5551043, -0.8225002],
    [1.5551043, -0.4050000],
    [1.9000000, -0.4050000],
    [1.9000000,  0.4050000]
])

# Create DataFrame
df = pd.DataFrame(data, columns=["r", "z"])

def parquet_to_netcdf(netcdf_file, headerdict):
    
    # Open a new NetCDF file for writing
    with Dataset(netcdf_file, 'w', format='NETCDF4') as ncfile:

        for key, value in headerdict.items():
            setattr(ncfile, key, value)

        lim_grp = ncfile.createGroup("limiter")  

        coord_dtype = np.dtype([("version", "<f8"),
                                ("r", "<f8", (37)),
                                ("z", "<f8", (37)),
                                ("phi_cut", "<f8")])

    
        lc = lim_grp.createCompoundType(coord_dtype, "LIMITER")

        # Dimension
        lim_grp.createDimension('singleDim', 1)

        version = headerdict["version"] + 0.1 * headerdict["revision"]

        var = lim_grp.createVariable("efit",
                                 lc,
                                 ("singleDim",))

        data = np.empty(1, lc.dtype_view)
        data["version"] = version
        data["r"] = df['r']
        data["z"] = df["z"]
        data["phi_cut"] = 0

        var[:] = data

if __name__ == "__main__":

    # Metadata for the netcdf file
    headerdict = {
        "Conventions": "",
        "device": "MAST",
        "class": "limiter",
        "system": "limiter",
        "configuration": "geometry",
        "shotRangeStart": 0,
        "shotRangeStop": 400000,
        "content": "geometry of the limiter for MAST",
        "comment": "",
        "units": "SI, m",
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
        "creatorCode": "python create_netcdf_limiter.py",
        "creationDate": datetime.strftime(datetime.now(), "%Y-%m-%d"),
        "createdBy": "jhodson",
        "testCode": "",
        "testDate": "",
        "testedBy": ""
    }

    # Example usage
    parquet_to_netcdf("limiter.nc", headerdict)