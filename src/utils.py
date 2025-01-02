from contextlib import contextmanager
import json
import sys
import uuid
from pathlib import Path

from distributed import get_client

from src.log import logger


@contextmanager
def nullcontext(enter_result=None):
    yield enter_result


def connected_to_cluster():
    try:
        get_client()
        return True
    except ValueError:
        return False


def harmonise_name(name: str) -> str:
    name = name.replace("/", "_")
    name = name.replace(" ", "_")
    name = name.replace("-", "_")
    name = name.replace("(", "")
    name = name.replace(")", "")
    name = name.replace(",", "")
    name = name.strip("_")
    name = name.strip("/")
    name = name.split("_", maxsplit=1)[-1]
    name = name.lower()
    return name


def get_uuid(name: str, shot: int) -> str:
    oid_name = f"{shot}/{name}"
    return str(uuid.uuid5(uuid.NAMESPACE_OID, oid_name))


def get_shot_list(args):
    """Get the list of shot numbers from the cli arguments"""

    if args.shot_file is not None:
        shot_list = read_shot_file(args.shot_file)
    elif args.shot_min is not None and args.shot_max is not None:
        shot_list = list(range(args.shot_min, args.shot_max + 1))
    elif args.shot is not None:
        shot_list = [args.shot]
    else:
        logger.error("One of --shot, --shot-file or --shot-min/max must be set.")
        sys.exit(-1)

    return shot_list


def read_shot_file(shot_file: str) -> list[int]:
    with open(shot_file) as f:
        shot_nums = f.readlines()[1:]
        shot_nums = map(lambda x: x.strip(), shot_nums)
        shot_nums = list(sorted(map(int, shot_nums)))
    return shot_nums


def read_json_file(file_name: str):
    with Path(file_name).open("r") as handle:
        return json.load(handle)
