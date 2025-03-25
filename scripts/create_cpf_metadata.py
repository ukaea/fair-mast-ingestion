import argparse
import multiprocessing as mp
from functools import partial

import pandas as pd
import pycpf
import requests

from src.core.log import logger


def read_cpf_for_shot(shot, columns):
    logger.info(f"Processing shot {shot}")

    API_URL = "http://uda2.hpc.l/cpf/api/query"
    columns = pycpf.columns()
    batch_size = 100
    cpf_data = {}

    for i in range(3):
        start = i * batch_size
        end = (i + 1) * batch_size
        column_batch = columns[start:end]

        params = [("columns", c[0]) for c in column_batch]
        params += [("filters", f"shot = {shot}")]

        response = requests.get(API_URL, params=params)
        response = response.json()
        response = response[str(shot)] if str(shot) in response else {}
        cpf_data.update(response)

    cpf_data["shot_id"] = shot
    logger.info(f"Done shot {shot}!")
    return cpf_data


def main():
    logger.setLevel("INFO")
    parser = argparse.ArgumentParser(
        prog="FAIR MAST Ingestor",
        description="Parse the MAST archive and writer to Zarr/NetCDF/HDF files",
    )

    parser.add_argument("name")
    parser.add_argument("--shot-min", type=int, default=None)
    parser.add_argument("--shot-max", type=int, default=None)
    args = parser.parse_args()

    name = args.name
    shot_ids = range(args.shot_min, args.shot_max + 1)

    columns = pycpf.columns()
    columns = pd.DataFrame(columns, columns=["name", "description"])
    columns.to_parquet(f"data/{name}_cpf_columns.parquet")

    pool = mp.Pool(16)
    column_names = columns["name"].values
    func = partial(read_cpf_for_shot, columns=column_names)
    mapper = pool.imap_unordered(func, shot_ids)
    rows = [item for item in mapper]
    cpf_data = pd.DataFrame(rows)

    # Convert objects to strings
    for column in cpf_data.columns:
        dtype = cpf_data[column].dtype
        if isinstance(dtype, object):
            cpf_data[column] = cpf_data[column].astype(str)

    cpf_data.to_parquet(f"data/{name}_cpf_data.parquet")
    print(cpf_data)


if __name__ == "__main__":
    main()
