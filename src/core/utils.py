import json
import subprocess
import sys
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from functools import lru_cache
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq
from distributed import get_client

from src.core.log import logger

_GIT_CWD = Path(__file__).resolve().parent
_REPO_URL = "https://github.com/ukaea/fair-mast-ingestion"

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


PARQUET_SUFFIXES = (".parquet", ".pq")


def read_shot_file(shot_file: str) -> list[int]:
    """Read the list of shot numbers from a file.

    The file can be a parquet file (a suffix of .parquet or .pq) or a delimited
    text file such as a CSV. The shot numbers must be in the first column. A
    header row is optional. Rows that do not contain a number are ignored.
    """
    path = Path(shot_file)
    if not path.exists():
        logger.error(f'No shot file exists called "{path}"')
        sys.exit(-1)

    if path.suffix.lower() in PARQUET_SUFFIXES:
        values = _read_parquet_column(path)
    else:
        values = pd.read_csv(path, header=None, usecols=[0]).iloc[:, 0]

    numbers = pd.to_numeric(values, errors="coerce")
    skipped = int(numbers.isna().sum())
    if skipped > 0:
        logger.debug(f"Ignored {skipped} row(s) without a shot number in {path}")

    shot_nums = sorted(int(number) for number in numbers.dropna())
    if len(shot_nums) == 0:
        logger.error(f'No shot numbers found in "{path}"')
        sys.exit(-1)

    return shot_nums


def _read_parquet_column(path: Path, index: int = 0) -> pd.Series:
    """Read a single column of a parquet file by its position."""
    name = pq.read_schema(path).names[index]
    return pd.read_parquet(path, columns=[name])[name]


def read_json_file(file_name: str):
    with Path(file_name).open("r") as handle:
        return json.load(handle)


def _run_git(args: list[str]) -> str | None:
    try:
        result = subprocess.run(
            ["git", *args],
            cwd=_GIT_CWD,
            capture_output=True,
            text=True,
            check=False,
            timeout=5,
        )
    except (FileNotFoundError, subprocess.TimeoutExpired) as e:
        logger.debug(f"git {' '.join(args)} failed: {e}")
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


@lru_cache(maxsize=1)
def get_commit_url() -> str | None:
    sha = _run_git(["rev-parse", "HEAD"])
    if not sha:
        return None
    suffix = " (dirty)" if _run_git(["status", "--porcelain", "--untracked-files=no"]) else ""
    return f"{_REPO_URL}/tree/{sha}{suffix}"


def get_ingestion_provenance() -> dict:
    info = {
        "ingested_at": datetime.now(timezone.utc)
        .replace(microsecond=0)
        .isoformat()
        .replace("+00:00", "Z")
    }
    commit_url = get_commit_url()
    if commit_url:
        info["commit_url"] = commit_url
    return info