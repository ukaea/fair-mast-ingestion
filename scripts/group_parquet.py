import argparse

import pyarrow.dataset as ds
import pyarrow.parquet as pq


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source_dir")
    parser.add_argument("output_file")
    args = parser.parse_args()

    # Read all parquet files in a directory
    dataset = ds.dataset(args.source_dir, format="parquet")

    # Convert to PyArrow Table
    table = dataset.to_table()

    # Save back to a single Parquet file
    pq.write_table(table, args.output_file)


if __name__ == "__main__":
    main()
