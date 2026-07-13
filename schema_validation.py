# schema_validation.py
import pandas as pd
from config import CLEAN_PARQUET, CORE_GASES

def validate_schema(df: pd.DataFrame) -> dict:
    """Kiểm tra schema cơ bản sau khi clean."""
    required_cols = ["transformer_id", "sample_day", "tested_day"] + CORE_GASES + ["temp", "water"]
    missing_cols = [col for col in required_cols if col not in df.columns]
    issues = []
    if missing_cols:
        issues.append(f"Missing columns: {missing_cols}")
    # Kiểm tra kiểu dữ liệu
    if not pd.api.types.is_datetime64_any_dtype(df["sample_day"]):
        issues.append("sample_day không phải datetime")
    for col in CORE_GASES:
        if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
            issues.append(f"{col} không phải numeric")
    # Kiểm tra dữ liệu không hợp lệ
    for col in CORE_GASES:
        if (df[col] < 0).any():
            issues.append(f"{col} chứa giá trị âm")
    return {"valid": len(issues)==0, "issues": issues}

if __name__ == "__main__":
    df = pd.read_parquet(CLEAN_PARQUET)
    result = validate_schema(df)
    print("Schema validation:", result)