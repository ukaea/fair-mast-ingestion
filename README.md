### FAIR MAST Data Ingestion

## Project Structure

Below is a brief overview of the project structure
```
|-- campaign_shots      # CSV lists of shots for each MAST campaign
|-- configs             # Config files for each level of ingestion
|-- geometry            # Geometry data files for each diagnostic source 
|-- jobs                # Job scripts for different HPC machines
|-- mappings            # Mapping files for transforming units, names, dimensions, etc.
|-- notebooks           # Notebooks for checking outputs
|-- scripts             # Misc scripts for metadata curation
|-- src                 # Source code for ingestion tools
|   |-- core            # Core modules for ingestion, shared between all levels
|   |-- level1          # Level1 data ingestion code
|   |-- level2          # Level2 data ingestion code
`-- tests               # Unit tests
    |-- core            # Core module unit tests
    |-- level1          # Level1 module unit tests
    |-- level2          # Level2 module unit tests
```

## Installation and Setup

Clone the repository and fetch data files (Git LFS must be installed):

```sh
git clone git@github.com:ukaea/fair-mast-ingestion.git
cd fair-mast-ingestion
git lfs pull
```

Create a new python virtual environment:

```sh
uv venv --python 3.12.6 
source .venv/bin/activate
```

Update pip and install required packages:

```sh
uv pip install git+ssh://git@git.ccfe.ac.uk/MAST-U/mastcodes.git@release/1.3.10#subdirectory=uda/python
uv pip install -e .
uv pip install -e ".[dev]"
uv pip install -e ".[mpi]"
```

If running on CSD3, we must also source the SSL certificate information by running the following command. Without this UDA cannot connect to the UKAEA network.

```sh
source ~/rds/rds-ukaea-ap002-mOlK9qn0PlQ/fairmast/uda-ssl.sh
```

Finally, for uploading to S3 we need to create a local config file with the bucket keys. Create a file called `.s5cfg.stfc` with the following information:

```
[default]
aws_access_key_id=<access-key>
aws_secret_access_key=<secret-key>
```

You should now be able to run the following commands.

## Submitting runs on CSD3

#### First Run on CSD3

This will ingest data into the test folder in S3. The small_ingest script allows you to put one file of shots into the ingestion.

1. First submit a job to collect all the metadata:

```sh
sbatch ./jobs/metadata.csd3.slurm.sh
```

2. Then submit an ingestion job

Argument 1 (e.g. s3://mast/test/shots/) is where the data will ingest to, and argument 2 is the file of shots to ingest (e.g. campaign_shots/tiny_campaign.csv), arguments 3 and greater are the sources (e.g. amc)

```sh
sbatch ./jobs/small_ingest.csd3.slurm.sh s3://mast/test/shots/ campaign_shots/tiny_campaign.csv amc
```

#### Ingesting All Shots

This ingestion job runs through all shots for the specified source (e.g. amc)

```sh
sbatch ./jobs/small_ingest.csd3.slurm.sh s3://mast/test/shots/ amc
```

## Manually Running Ingestor

### Local Ingestion

The following section details how to ingest data into a local folder on freia with UDA.

1. Parse the metadata for all signals and sources for a list of shots with the following command

```sh
mpirun -n 16 python3 -m src.create_uda_metadata data/uda campaign_shots/tiny_campaign.csv 
```

```sh
mpirun -np 16 python3 -m src.main data/local campaign_shots/tiny_campaign.csv --metadata_dir data/uda --source_names amc xsx --file_format nc
```

Files will be output in the NetCDF format to `data/local`.

### Ingestion to S3

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

## CPF Metadata

To parse CPF metadata we can use the following script (only on Friea):

```sh
qsub ./jobs/freia_write_cpf.qsub campaign_shots/tiny_campaign.csv
```

