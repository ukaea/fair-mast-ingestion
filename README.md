### FAIR MAST Data Ingestion

## Running on CSD3
### Installation on CSD3

After logging into your CSD3 account (on Icelake node), first load the correct Python module:

```sh
module load python/3.9.12/gcc/pdcqf4o5
```

Clone the repository:

```sh
git clone git@github.com:ukaea/fair-mast-ingestion.git
cd fair-mast-ingestion
```

Create a virtual environment:

```sh
python -m venv fair-mast-ingestion
source fair-mast-ingestion/bin/activate
```

Update pip and install required packages:

```sh
python -m pip install --U pip
python -m pip install -e .
```

The final step to installation is to have mastcodes:

```sh
git clone git@git.ccfe.ac.uk:MAST-U/mastcodes.git
cd mastcodes
```

Edit `uda/python/setup.py` and change the "version" to 1.3.9.

```sh
python -m pip install uda/python
source ~/rds/rds-ukaea-mast-sPGbyCAPsJI/uda-ssl.sh
```

You should now be able to run the following commands.

### Submitting runs on CSD3

1. First submit a job to collect all the metadata:

```sh
sbatch ./jobs/metadata.csd3.slurm.sh
```

2. Then submit an ingestion job

```sh
sbatch ./jobs/ingest.csd3.slurm.sh campaign_shots/tiny_campaign.csv s3://mast/test/shots/ amc
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

