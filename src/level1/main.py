import argparse

from src.core.config import load_config
from src.core.log import logger
from src.core.utils import get_shot_list
from src.level1.workflow import IngestionWorkflow, WorkflowManager


def main():
    parser = argparse.ArgumentParser(
        prog="FAIR MAST Ingestor",
        description="Write Tokamak data to Zarr/NetCDF/HDF files",
    )

    parser.add_argument("-o", "--output-path", default="./")
    parser.add_argument("-v", "--verbose", action="store_true")
    parser.add_argument("-n", "--n-workers", type=int, default=None)
    parser.add_argument("--shot", type=int)
    parser.add_argument("--shot-min", type=int)
    parser.add_argument("--shot-max", type=int)
    parser.add_argument("--shot-file", type=str)
    parser.add_argument("--config-file", type=str, default="./config.yml")
    parser.add_argument("-i", "--include-sources", nargs="+", default=[])
    parser.add_argument("-e", "--exclude-sources", nargs="+", default=[])
    parser.add_argument("--file_format", choices=["zarr", "nc", "h5"], default="zarr")
    parser.add_argument("--facility", choices=["MAST", "MASTU"], default="MAST")

    args = parser.parse_args()
    if args.verbose:
        logger.setLevel("DEBUG")

    shot_list = get_shot_list(args)
    config = load_config(args.config_file)

    workflow = IngestionWorkflow(
        config=config,
        facility=args.facility,
        include_sources=args.include_sources,
        exclude_sources=args.exclude_sources,
        verbose=args.verbose,
    )

    workflow_manager = WorkflowManager(workflow)
    workflow_manager.run_workflows(shot_list, args.n_workers)


if __name__ == "__main__":
    main()
