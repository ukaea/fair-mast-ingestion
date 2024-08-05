### FAIR MAST Data Ingestion

## Installation

```sh
pip install -e .
```

## Local Ingestion

The following section details how to ingest data into a local folder on freia with UDA.

1. Parse the metadata for all signals and sources for a list of shots with the following command

```sh
mpirun -n 16 python3 -m src.create_uda_metadata data/uda campaign_shots/tiny_campaign.csv 
```

```sh
mpirun -np 16 python3 -m src.main data/local campaign_shots/tiny_campaign.csv --metadata_dir data/uda --source_names amc xsx --file_format nc
```

Files will be output in the NetCDF format to `data/local`.

## Ingestion to S3

The following section details how to ingest data into the s3 storage on freia with UDA.

1. Parse the metadata for all signals and sources for a list of shots with the following command

```sh
mpirun -n 16 python3 -m src.create_uda_metadata data/uda campaign_shots/tiny_campaign.csv 
```

This will create the metadata for the tiny campaign. You may do the same for full campaigns such as `M9`.

2. Run the ingestion pipleline by submitting the following job:

```sh
mpirun -np 16 python3 -m src.main data/local campaign_shots/tiny_campaign.csv --bucket_path s3://mast/test/shots --source_names amc xsx --file_format zarr --upload --force
```

This will submit a job to the freia job queue that will ingest all of the shots in the tiny campaign and push them to the s3 bucket.
