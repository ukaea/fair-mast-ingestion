import pandas as pd
import pytest

from src.core.utils import read_shot_file


def write_csv(path, text: str):
    path.write_text(text)
    return path


def test_read_shot_file_csv_with_header(tmp_path):
    file_name = write_csv(tmp_path / "shots.csv", "shot_id\n30350\n30351\n")
    assert read_shot_file(file_name) == [30350, 30351]


def test_read_shot_file_csv_without_header(tmp_path):
    file_name = write_csv(tmp_path / "shots.csv", "30350\n30351\n")
    assert read_shot_file(file_name) == [30350, 30351]


def test_read_shot_file_csv_uses_first_column(tmp_path):
    file_name = write_csv(
        tmp_path / "shots.csv", "shot_id,campaign\n30351,M9\n30350,M9\n"
    )
    assert read_shot_file(file_name) == [30350, 30351]


def test_read_shot_file_sorts_the_shots(tmp_path):
    file_name = write_csv(tmp_path / "shots.csv", "shot_id\n30351\n11695\n30350\n")
    assert read_shot_file(file_name) == [11695, 30350, 30351]


def test_read_shot_file_parquet(tmp_path):
    file_name = tmp_path / "shots.parquet"
    frame = pd.DataFrame({"shot_id": [30351, 30350], "campaign": ["M9", "M9"]})
    frame.to_parquet(file_name)
    assert read_shot_file(file_name) == [30350, 30351]


def test_read_shot_file_missing_file(tmp_path):
    with pytest.raises(SystemExit):
        read_shot_file(tmp_path / "missing.csv")


def test_read_shot_file_without_shots(tmp_path):
    file_name = write_csv(tmp_path / "shots.csv", "shot_id\n")
    with pytest.raises(SystemExit):
        read_shot_file(file_name)
