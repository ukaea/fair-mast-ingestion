import argparse
import numpy as np
import pandas as pd
import multiprocessing as mp
from functools import partial
from pathlib import Path
from rich.progress import track
from pycpf import pycpf


def read_cpf_for_shot(shot, columns):
    cpf_data = {}
    for name in columns:
        entry = pycpf.query(name, f"shot = {shot}") 
        value = entry[name][0] if name in entry else np.nan
        cpf_data[name] = value 

    cpf_data['shot_id'] = shot
    return cpf_data

def main():
    parser = argparse.ArgumentParser(
        prog="FAIR MAST Ingestor",
        description="Parse the MAST archive and writer to Zarr/NetCDF/HDF files",
    )

    parser.add_argument("shot_file")
    args = parser.parse_args()

    shot_file = args.shot_file
    shot_ids = pd.read_csv(shot_file)
    shot_ids = shot_ids['shot_id'].values

    columns = pycpf.columns()
    columns = pd.DataFrame(columns, columns=['name', 'description'])
    columns.to_parquet(f'data/{Path(shot_file).stem}_cpf_columns.parquet')

    pool = mp.Pool(16)
    column_names = columns['name'].values
    func = partial(read_cpf_for_shot, columns=column_names)
    mapper = pool.imap_unordered(func, shot_ids)
    rows = [item for item in track(mapper, total=len(shot_ids))]
    cpf_data = pd.DataFrame(rows)

    # Convert objects to strings
    for column in cpf_data.columns:
        dtype = cpf_data[column].dtype
        if isinstance(dtype, object):
            cpf_data[column] = cpf_data[column].astype(str)

    cpf_data.to_parquet(f'data/{Path(shot_file).stem}_cpf_data.parquet')
    print(cpf_data)
   

if __name__ == "__main__":
    main()
