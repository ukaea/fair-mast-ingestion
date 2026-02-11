# Level 2 Ingestion Documentation

This document provides comprehensive instructions for running the Level 2 data ingestion pipeline for FAIR MAST data.

## Overview

Level 2 ingestion processes raw data and produces standardized, interpolated datasets stored in Zarr format. The pipeline supports both MAST and MAST-U facilities.

---

## 1. Running CPF Metadata Job on Freia

The CPF (Central Physics Files) metadata job creates metadata files for all shots in the specified shot ranges.

### Prerequisites
- Access to Freia cluster
- Access to CSD3 cluster
- Virtual environment activated with required dependencies
- Shot range information for MAST and MAST-U
- `s5cmd` installed and configured on your system

### Submitting the Job

Navigate to the project directory and submit the job using:

```bash
qsub jobs/cpf_metadata.freia.qsub
```

### Job Configuration

The job is configured in [jobs/cpf_metadata.freia.qsub](jobs/cpf_metadata.freia.qsub):

- **Parallel Environment**: 16 MPI processes
- **Time Limit**: 48 hours
- **Job Name**: `fairmast-cpf-writer`

### What It Does

The script runs two metadata creation commands:

1. **MAST**: Processes shots 11695 to 30475
   ```bash
   python3 -m scripts.create_cpf_metadata mast --shot-min 11695 --shot-max 30475
   ```

2. **MAST-U**: Processes shots 41139 to 51056
   ```bash
   python3 -m scripts.create_cpf_metadata mastu --shot-min 41139 --shot-max 51056
   ```

### Monitoring

Check job status:
```bash
qstat -u $USER
```

View output logs in the current working directory:
```bash
tail -f fairmast-cpf-writer.o<job_id>
```

---

## 2.1 Running Level 2 Ingestion CLI Tools Locally

To ingest UDA data, the Level 2 ingestion tools can be run locally for testing or processing small batches of shots.

### Command Structure

```bash
python3 -m src.level2.main <mapping_file> [OPTIONS]
```

### Required Arguments

- `mapping_file`: Path to the YAML mapping file (e.g., `mappings/level2/mast.yml` or `mappings/level2/mastu.yml`)

### Optional Arguments

| Flag | Description | Default |
|------|-------------|---------|
| `-c, --config-file` | Configuration file path | `./configs/level2.yml` |
| `--shot` | Process a single shot | None |
| `--shot-min` | Minimum shot number (requires --shot-max) | None |
| `--shot-max` | Maximum shot number (requires --shot-min) | None |
| `--shots` | Space-separated list of specific shots | None |
| `--dt` | Time delta for interpolation | 0.00025 |
| `-i, --include-datasets` | Only process specified datasets | All |
| `-e, --exclude-datasets` | Exclude specified datasets | None |
| `-v, --verbose` | Enable debug logging | False |
| `-o, --output-path` | Override output directory | From config |
| `-n, --n-workers` | Number of parallel workers | System default |

### Example Commands

**Process a single shot:**
```bash
python3 -m src.level2.main mappings/level2/mast.yml --shot 30350
```

**Process a range of shots:**
```bash
python3 -m src.level2.main mappings/level2/mast.yml \
    --shot-min 11695 --shot-max 11700
```

**Process specific shots:**
```bash
python3 -m src.level2.main mappings/level2/mast.yml \
    --shots 11889 30359 12165 30377
```

**Process with multiple workers:**
```bash
python3 -m src.level2.main mappings/level2/mast.yml \
    -n 8 --shot-min 11695 --shot-max 11700
```

**Exclude specific datasets:**
```bash
python3 -m src.level2.main mappings/level2/mast.yml \
    --shot 30350 -e camera_visible
```

**Custom output path:**
```bash
python3 -m src.level2.main mappings/level2/mast.yml \
    --shot 30350 -o /path/to/output
```

### Configuration Files

**Local Config** ([configs/level2.yml](configs/level2.yml)):
- Uses Zarr format 2
- Outputs to `/tmp/fair-mast/level2`
- Reads from UDA and S3-hosted Level 1 data

**CSD3 Config** ([configs/level2.csd3.yml](configs/level2.csd3.yml)):
- Uses Zarr format 3
- Outputs to CSD3 project directory
- Optimized for cluster environment

---

## 2.2 Using the SLURM Job File for CSD3

For large-scale processing on the CSD3 cluster, use the SLURM job submission script.

### Prerequisites
- Access to CSD3 cluster
- Project allocation: `UKAEA-AP002-CPU`
- Virtual environment configured
- UDA SSL configuration file

### Job Configuration

The job is configured in [jobs/ingest.level2.csd3.slurm.sh](jobs/ingest.level2.csd3.slurm.sh):

- **Partition**: `ukaea-icl`
- **Time Limit**: 36 hours
- **Memory**: 250 GB
- **Tasks**: 64 (8 nodes × 8 tasks)
- **Output**: `fair-mast-ingest_<job_id>.out`

### Submitting the Job

```bash
sbatch jobs/ingest.level2.csd3.slurm.sh
```

### Customizing the Job

Edit [jobs/ingest.level2.csd3.slurm.sh](jobs/ingest.level2.csd3.slurm.sh) to modify:

1. **Shot range**: Uncomment and modify the range-based command
   ```bash
   mpirun -n $num_workers \
       python3 -m src.level2.main mappings/level2/mast.yml \
       -c ./configs/level2.csd3.yml \
       --shot-min 11695 --shot-max 30474 \
       -e camera_visible -o $output_dir
   ```

2. **Specific shots**: Modify the shots list in the current command
   ```bash
   mpirun -n $num_workers \
       python3 -m src.level2.main mappings/level2/mast.yml \
       -c ./configs/level2.csd3.yml \
       --shots 11889 30359 12165 30377 30350 ... \
       -e camera_visible -o $output_dir
   ```

3. **Output directory**: Change the `output_dir` variable
   ```bash
   output_dir="/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2/"
   ```

4. **Facility**: Switch between MAST and MAST-U
   ```bash
   python3 -m src.level2.main mappings/level2/mastu.yml ...
   ```

### Environment Setup

The script automatically:
1. Activates the Python virtual environment
2. Sources UDA SSL configuration from `/rds/project/rds-mOlK9qn0PlQ/fairmast/uda-ssl.sh`
3. Uses MPI for parallel processing across nodes

### Monitoring

**Check job status:**
```bash
squeue -u $USER
```

**View job details:**
```bash
scontrol show job <job_id>
```

**Monitor output in real-time:**
```bash
tail -f fair-mast-ingest_<job_id>.out
```

**Cancel job:**
```bash
scancel <job_id>
```

---

## 3. Uploading Data to S3 Using upload.csd3.sh

After ingestion completes, upload the processed data to S3 storage using the provided upload script.

### Prerequisites
- Completed Level 2 ingestion with data in the output directory
- S3 credentials file: `.s5cfg.stfc`
- `s5cmd` tool installed
- Access to STFC S3 storage

### Job Configuration

The upload job is configured in [jobs/upload.csd3.sh](jobs/upload.csd3.sh):

- **Partition**: `ukaea-icl`
- **Time Limit**: 36 hours
- **Memory**: 250 GB
- **Tasks**: 1 (single node)

### S3 Configuration

- **Credentials**: `.s5cfg.stfc` (must be in project root)
- **Endpoint**: `https://s3.echo.stfc.ac.uk`
- **Local Path**: `/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2/`
- **Remote Path**: `s3://mast/level2/shots/`

### Submitting the Upload Job

```bash
sbatch jobs/upload.csd3.sh
```

### What It Does

The script performs two operations:

1. **Initial Copy**: Copies all files to S3 with public-read ACL
   ```bash
   s5cmd --credentials-file .s5cfg.stfc \
         --endpoint-url https://s3.echo.stfc.ac.uk \
         cp --acl public-read $local_path $remote_path
   ```

2. **Sync**: Synchronizes changes and removes deleted files
   ```bash
   s5cmd --credentials-file .s5cfg.stfc \
         --endpoint-url https://s3.echo.stfc.ac.uk \
         sync --delete --acl public-read $local_path $remote_path
   ```

### Customizing Upload Paths

Edit [jobs/upload.csd3.sh](jobs/upload.csd3.sh) to modify:

**Local path** (source):
```bash
local_path=/rds/project/rds-mOlK9qn0PlQ/fairmast/upload-tmp/level2/
```

**Remote path** (destination):
```bash
remote_path=s3://mast/level2/shots/
```

**Credentials file**:
```bash
credential=.s5cfg.stfc
```

**Endpoint URL** (for different S3 providers):
```bash
endpoint=https://s3.echo.stfc.ac.uk
```

### Monitoring Upload

**Check job status:**
```bash
squeue -u $USER
```

**View upload progress:**
```bash
tail -f fair-mast-ingest-upload_<job_id>.out
```

### Verifying Upload

After upload completes, verify the data is accessible:

```bash
# Using s5cmd
s5cmd --credentials-file .s5cfg.stfc \
      --endpoint-url https://s3.echo.stfc.ac.uk \
      ls s3://mast/level2/shots/
```

### Upload Options

The `s5cmd` commands include:
- `--acl public-read`: Makes uploaded data publicly readable
- `--delete`: Removes remote files not present locally (sync only)
- `cp`: Copies files without removing remote-only files
- `sync`: Ensures remote exactly matches local

---


## 4. Building Metadata Index with s3_metadata_csd3.slurm

After uploading data to S3, build the metadata index to enable efficient querying and discovery of the ingested shots.

### Prerequisites
- Completed S3 upload with data in `s3://mast/level2/shots/`
- S3 credentials file: `.s5cfg.stfc`
- Access to CSD3 cluster
- Virtual environment configured

### Job Configuration

The metadata indexing job is configured in [jobs/s3_metadata_csd3.slurm](jobs/s3_metadata_csd3.slurm):

- **Partition**: `ukaea-icl`
- **Time Limit**: 12 hours
- **Memory**: 128 GB
- **Tasks**: 1 (single node)

### Submitting the Job

```bash
sbatch jobs/s3_metadata_csd3.slurm
```

### What It Does

The script scans the S3 bucket and creates a comprehensive metadata index that includes:
- Available shots
- Dataset information
- Temporal coverage
- File locations
- Data dimensions

### Monitoring

**Check job status:**
```bash
squeue -u $USER
```

**View indexing progress:**
```bash
tail -f s3-metadata-builder_<job_id>.out
```

### Output

The metadata index is written to:
- **S3 Location**: `s3://mast/level2/metadata/`
- **Local Cache**: May be stored in project directory for verification

### Verifying the Index

After completion, verify the metadata index exists:

```bash
s5cmd --credentials-file .s5cfg.stfc \
    --endpoint-url https://s3.echo.stfc.ac.uk \
    ls s3://mast/level2/metadata/
```

The final output files will be called:

```sh
./mast-level2-sources.parquet
./mast-level2-signals.parquet
```

In the local directory where the job was run. These files can be used to update the metadata service for FAIR MAST.

---

## Complete Workflow Example

### Full Pipeline on CSD3

1. **Generate CPF metadata** (on Freia):
   ```bash
   qsub jobs/cpf_metadata.freia.qsub
   ```

2. **Run Level 2 ingestion** (on CSD3):
   ```bash
   sbatch jobs/ingest.level2.csd3.slurm.sh
   ```

3. **Upload to S3** (on CSD3):
   ```bash
   sbatch jobs/upload.csd3.sh
   ```

### Local Testing Workflow

1. **Test single shot locally**:
   ```bash
   python3 -m src.level2.main mappings/level2/mast.yml \
       --shot 30350 -v
   ```

2. **Verify output**:
   ```bash
   ls -lh /tmp/fair-mast/level2/
   ```

3. **Process production data on cluster** (after testing succeeds)

---
