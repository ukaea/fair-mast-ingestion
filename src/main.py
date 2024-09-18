import argparse
import logging
from functools import partial
from dask_mpi import initialize
import lakefs
from src.lake_fs import lakefs_merge_into_main, create_branch
from src.uploader import UploadConfig, LakeFSUploadConfig
from src.workflow import LakeFSIngestionWorkflow, LocalIngestionWorkflow, WorkflowManager
from src.utils import read_shot_file

def main():

    initialize()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="FAIR MAST Ingestor",
        description="Parse the MAST archive and writer to Zarr/NetCDF/HDF files",
    )

    parser.add_argument("dataset_path")
    parser.add_argument("shot_file")
    parser.add_argument("--credentials_file", default="lakectl.cfg")
    parser.add_argument("--serial", default=False, action='store_true')
    parser.add_argument("--endpoint_url", default="http://localhost:8000")
    parser.add_argument("--upload", nargs='?', const=False, default=False)
    parser.add_argument("--metadata_dir", default="data/uda")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--signal_names", nargs="*", default=[])
    parser.add_argument("--source_names", nargs="*", default=[])
    parser.add_argument("--file_format", choices=['zarr', 'nc', 'h5'], default='zarr')
    parser.add_argument("--facility", choices=['MAST', 'MASTU'], default='MAST')

    args = parser.parse_args()
    if args.upload:
        new_branch = create_branch(args.upload)
        config = LakeFSUploadConfig(
            credentials_file=args.credentials_file,
            endpoint_url=args.endpoint_url,
            repository=args.upload,
            branch=new_branch
        )
        workflow_cls = partial(LakeFSIngestionWorkflow, upload_config=config)
    else:
        config = None
        workflow_cls = LocalIngestionWorkflow

    shot_list = read_shot_file(args.shot_file)

    for source in args.source_names:
        logging.info("------------------------")
        logging.info(f"Starting source {source}")

        workflow = workflow_cls(
            args.metadata_dir,
            args.dataset_path,
            force=args.force,
            signal_names=args.signal_names,
            source_names=[source],
            file_format=args.file_format,
            facility=args.facility
        )

        workflow_manager = WorkflowManager(workflow)
        workflow_manager.run_workflows(shot_list, parallel=not args.serial)
        logging.info(f"Finished source {source}")

    if args.upload:
        lakefs_merge_into_main(args.upload, new_branch)

if __name__ == "__main__":
    main()
