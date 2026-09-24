import pandas as pd
from services.profiling.schema_detector import SchemaDetector
from services.profiling.profiler import DataProfiler


def test_schema_detector_type_inference():
    df = pd.DataFrame({
        "customer_id": [f"CUST-{i}" for i in range(100)],
        "age": [25, 30, 45, 22, 60] * 20,
        "is_active": [True, False, True, True, False] * 20,
        "signup_date": ["2026-01-01", "2026-02-15", "2026-03-10", "2026-04-05", "2026-05-12"] * 20,
        "tier": ["Gold", "Silver", "Bronze", "Gold", "Platinum"] * 20,
        "notes": ["Customer reported positive feedback during initial onboarding survey."] * 100
    })

    schema = SchemaDetector.detect_schema(df)
    cols = schema["columns"]

    assert cols["customer_id"]["inferred_type"] == "IDENTIFIER"
    assert cols["age"]["inferred_type"] == "NUMERICAL"
    assert cols["is_active"]["inferred_type"] == "BOOLEAN"
    assert cols["signup_date"]["inferred_type"] == "DATETIME"
    assert cols["tier"]["inferred_type"] == "CATEGORICAL"
    assert cols["notes"]["inferred_type"] == "TEXT"


def test_data_profiler_statistics():
    df = pd.DataFrame({
        "salary": [50000, 60000, 75000, 90000, 150000],
        "department": ["Sales", "Engineering", "Engineering", "HR", "Sales"]
    })

    profile = DataProfiler.profile_dataset(df)
    salary_prof = profile["columns"]["salary"]
    dept_prof = profile["columns"]["department"]

    assert salary_prof["mean"] == 85000.0
    assert salary_prof["min"] == 50000.0
    assert salary_prof["max"] == 150000.0
    assert "histogram" in salary_prof
    assert len(dept_prof["top_categories"]) == 3
