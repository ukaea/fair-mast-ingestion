### FAIR MAST Data Ingestion

## Project Structure

Below is a brief overview of the project structure
```
|-- campaign_shots      # CSV lists of shots for each MAST campaign
|-- configs             # Config files for each level of ingestion
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

### UDA SSL Configuration

A prerequisite for accessing MAST and MAST-U data is to have a SSL certificate for UDA. <br>
- First mint a certificate: [pkiuda](https://pkiuda.ukaea.uk/). <br>
- Then copy it to your home directory on CSD3.
- Set it up in your CSD3 environment as described [here](https://ukaea.github.io/UDA/authentication/#configuring-an-authenticated-client-connection).

You must be given the permissions to be able to mint a certificate and be on the UKAEA VPN/internal network.

```sh
source ~/.uda-ssl.sh
```

### Uploading to S3 Config

Finally, for uploading to S3 we need to create a local config file with the bucket keys. Create a file called `.s5cfg.stfc` with the following information:

```
[default]
aws_access_key_id=<access-key>
aws_secret_access_key=<secret-key>
```


## Running Ingestion

See `INGESTION.md`.
