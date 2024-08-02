## Ingestion to S3

The following section details how to ingest data into the s3 storage on freia with UDA.

1. SSH onto freia and setup a local development environment following the instuctions above.
2. Parse the metadata for all signals and sources for a list of shots with the following command

```sh
mpirun -n 16 python3 -m src.archive.create_uda_metadata data/uda campaign_shots/tiny_campaign.csv 
```

This will create the metadata for the tiny campaign. You may do the same for full campaigns such as `M9`.

3. Run the ingestion pipleline by submitting the following job:

```sh
qsub ./jobs/freia_write_datasets.qsub campaign_shots/tiny_campaign.csv s3://mast/level1/shots
```

This will submit a job to the freia job queue that will ingest all of the shots in the tiny campaign and push them to the s3 bucket.
