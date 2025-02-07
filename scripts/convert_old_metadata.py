from pathlib import Path

import pandas as pd
from sqlalchemy import create_engine


def main():
    db_name = "./old-level1.db"
    db_name = Path(db_name).absolute()
    uri = f"sqlite:////{db_name}"
    con = create_engine(uri)

    sources = pd.read_parquet("old-data/data/index/sources.parquet")
    source_columns = ["uuid", "shot_id", "name", "description", "quality"]
    sources = sources[source_columns]
    sources["imas"] = None

    sources.to_sql("sources", con, index=False)

    signals = pd.read_parquet("old-data/data/index/signals.parquet")
    signal_columns = [
        "uuid",
        "shot_id",
        "name",
        "source",
        "description",
        "quality",
        "rank",
        "shape",
        "dimensions",
    ]
    signals = signals[signal_columns]

    signals["shape"] = signals["shape"].map(lambda x: ",".join(list(map(str, x))))
    signals["dimensions"] = signals["dimensions"].map(lambda x: ",".join(list(x)))
    signals["imas"] = None
    signals["units"] = ""

    signals.to_sql("signals", con, index=False)


if __name__ == "__main__":
    main()
