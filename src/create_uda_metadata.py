import argparse
import logging

from dask_mpi import initialize

from src.utils import read_shot_file
from src.workflow import MetadataWorkflow, WorkflowManager


def main():
    initialize()
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="UDA Archive Metadata Parser",
        description="Read metadata for UDA for all the signals and sources",
    )

    parser.add_argument("dataset_path")
    parser.add_argument("shot_file")

    args = parser.parse_args()

    shot_list = read_shot_file(args.shot_file)

    workflow = MetadataWorkflow(args.dataset_path)

    workflow_manager = WorkflowManager(workflow)
    workflow_manager.run_workflows(shot_list)


if __name__ == "__main__":
    main()
