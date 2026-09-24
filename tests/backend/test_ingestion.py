import io
import os
import pandas as pd
import pytest
from services.ingestion.csv_adapter import CSVAdapter
from services.ingestion.json_adapter import JSONAdapter
from services.ingestion.parquet_adapter import ParquetAdapter
from services.ingestion.factory import IngestionFactory
from services.ingestion.secure_upload import calculate_sha256, sanitize_filename


def test_csv_adapter_detection_and_load(tmp_path):
    csv_file = tmp_path / "test_semicolon.csv"
    csv_file.write_text("name;age;salary\nAlice;30;75000\nBob;45;120000\nCharlie;22;50000", encoding="utf-8")

    adapter = CSVAdapter()
    assert adapter.validate(str(csv_file)) is True

    df = adapter.load(str(csv_file))
    assert len(df) == 3
    assert list(df.columns) == ["name", "age", "salary"]
    assert df.iloc[0]["name"] == "Alice"

    meta = adapter.get_metadata(str(csv_file))
    assert meta["row_count"] == 3
    assert meta["column_count"] == 3
    assert meta["delimiter"] == ";"


def test_json_adapter_load(tmp_path):
    json_file = tmp_path / "records.json"
    json_file.write_text('[{"id": 1, "val": "A"}, {"id": 2, "val": "B"}]', encoding="utf-8")

    adapter = JSONAdapter()
    assert adapter.validate(str(json_file)) is True

    df = adapter.load(str(json_file))
    assert len(df) == 2
    assert "val" in df.columns


def test_parquet_adapter_load(tmp_path):
    parquet_file = tmp_path / "test.parquet"
    df_orig = pd.DataFrame({"x": [1, 2, 3], "y": [4.0, 5.5, 6.2]})
    df_orig.to_parquet(parquet_file)

    adapter = ParquetAdapter()
    assert adapter.validate(str(parquet_file)) is True

    df_loaded = adapter.load(str(parquet_file))
    assert len(df_loaded) == 3
    assert list(df_loaded.columns) == ["x", "y"]


def test_ingestion_factory_resolution():
    assert isinstance(IngestionFactory.get_adapter("csv"), CSVAdapter)
    assert isinstance(IngestionFactory.get_adapter("data.xlsx"), type(IngestionFactory.get_adapter("xlsx")))
    assert isinstance(IngestionFactory.get_adapter("test.json"), JSONAdapter)
    assert isinstance(IngestionFactory.get_adapter("sample.parquet"), ParquetAdapter)

    with pytest.raises(ValueError):
        IngestionFactory.get_adapter("unsupported.zip")


def test_secure_upload_helpers(tmp_path):
    f = tmp_path / "hello.txt"
    f.write_text("NEXUS Data Intelligence", encoding="utf-8")
    
    checksum = calculate_sha256(str(f))
    assert len(checksum) == 64

    clean = sanitize_filename("../../../malicious_file!!@@.csv")
    assert ".." not in clean
    assert clean.endswith(".csv")
