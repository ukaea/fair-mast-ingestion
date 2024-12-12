import logging

import pandas as pd
import s3fs
import xarray as xr
from dask.distributed import Client, as_completed
from dask_mpi import initialize

logging.basicConfig(level=logging.INFO)

def get_source(row, fs):
    dataset = xr.open_dataset(fs.get_mapper(row.url), engine='zarr')

    metadata = {
        'shot_id': row.shot_id,
        'source': row['name'],
        'url': row.url,
        'quality': row.quality,
        'nbytes': dataset.nbytes,
        'num_signals': len(dataset.data_vars)
    }
    return metadata


def main():
    initialize()
    client = Client()

    shot_df = pd.read_parquet('https://mastapp.site/parquet/shots')
    sources_df = pd.read_parquet('https://mastapp.site/parquet/sources')

    fs = s3fs.S3FileSystem(anon=True, endpoint_url="https://s3.echo.stfc.ac.uk")

    tasks = []
    for _, row in sources_df.iterrows():
        task = client.submit(get_source, row, fs)
        tasks.append(task)

    metadatas = []
    n = len(tasks)
    for i, task in enumerate(as_completed(tasks)):
        metadata = task.result()
        logging.info(f"Finished {metadata['url']} - {i+1}/{n} - {(i+1)/n*100:.2f}%")
        shot_info : pd.DataFrame = shot_df.loc[metadata['shot_id'] == shot_df.shot_id].iloc[0]
        shot_info = shot_info.to_dict()
        metadata.update(shot_info)
        metadatas.append(metadata)

    metadatas = pd.DataFrame(metadatas)
    metadatas.to_parquet('summary.parquet')

        

if __name__ == "__main__":
    main()