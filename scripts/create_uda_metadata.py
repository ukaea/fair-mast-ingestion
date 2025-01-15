import argparse
import logging
from pathlib import Path

import pandas as pd

from src.core.load import MissingMetadataError, MissingProfileError, UDALoader
from src.core.log import logger
from src.core.workflow_manager import WorkflowManager


def write_dataset(shot_num: int, output_dir: str):
    loader = UDALoader()
    try:
        infos = loader.get_signal_infos(shot_num)
    except MissingMetadataError:
        return

    infos = [info.model_dump() for info in infos]

    final_infos = []
    for info in infos:
        name = info["name"]
        logger.info(f"Loading {name}")
        try:
            item = loader.load(shot_num, name)
            info["dims"] = item.sizes.keys()
            info["sizes"] = item.sizes.values()
            final_infos.append(info)
        except MissingProfileError:
            continue

    infos = pd.DataFrame(final_infos)
    print(infos)
    file_name = Path(output_dir) / f"{shot_num}.parquet"
    infos.to_parquet(file_name)


def main():
    logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(
        prog="UDA Archive Metadata Parser",
        description="Read metadata for UDA for all the signals and sources",
    )

    parser.add_argument("--output-dir", type=str)
    parser.add_argument("--shot-min", type=int)
    parser.add_argument("--shot-max", type=int)
    parser.add_argument("--n_workers", type=int, default=8)

    args = parser.parse_args()

    shots = list(reversed(range(args.shot_min, args.shot_max + 1)))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    workflow_manager = WorkflowManager(write_dataset)
    workflow_manager.run_workflows(shots, args.n_workers, output_dir=output_dir)


if __name__ == "__main__":
    main()
