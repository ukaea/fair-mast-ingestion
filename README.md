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

Clone the repository and fetch data files:

```sh
git clone git@github.com:ukaea/fair-mast-ingestion.git
cd fair-mast-ingestion
```

Create a new python virtual environment:

```sh
uv venv --python 3.12.6 
source .venv/bin/activate
```

Update pip and install required packages:

```sh
uv pip install git+ssh://git@git.ccfe.ac.uk/MAST-U/mastcodes.git#subdirectory=uda/python
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


## Running Ingestion

The following section details how to ingest data into a local folder with UDA. 

First you must edit both the config files in `./configs/` to point the writer `output_path` at a sensible location:

```yaml
...
writer:
  type: "zarr"
  options:
    zarr_format: 2
    output_path: "/common/tmp/sjackson/upload-tmp/zarr/level1"
...
```

### Level 1 Ingestion

Below gives an example of running a level 1 ingestion which will write `ayc` data for shot `30421` from MAST.

```sh
mpirun -n 4 python3 -m src.level1.main -v --facility MAST --shot 30421 -i ayc
```

### Level 2 Ingestion

Below gives an example of running a level 2 ingestion which will write `thomson_scattering` data for shot `30421` from MAST.
```sh
mpirun -n 4 python3 -m src.level2.main mappings/level2/mast.yml -v --shot 30421 -i thomson_scattering
```

### Ingestion to S3

To ingest to S3 you must edit the config files in `./configs` to include the upload entry. You must specify the endpoint url and location to upload data to.
For example the following config sets the base path and endpoint url for object storage at CSD3:

```yaml
upload:
  base_path: "s3://mast/test/level1/shots"
  mode: 's5cmd'
  credentials_file: ".s5cfg.csd3"
  endpoint_url: "https://object.arcus.openstack.hpc.cam.ac.uk"
```

Then simple rerun the commands as above.

## CPF Metadata

To parse CPF metadata we can use the following script (only on Friea):

```sh
qsub ./jobs/freia_write_cpf.qsub campaign_shots/tiny_campaign.csv
```

