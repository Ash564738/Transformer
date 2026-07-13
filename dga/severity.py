# severity.py
import numpy as np
import pandas as pd

def calculate_severity(row, df_full=None):
    """Tính điểm severity cho một record, dựa trên ngưỡng IEEE C57.104 tạm."""
    score = 0
    # Trọng số ví dụ
    gas_weights = {"h2": 1, "ch4": 1, "c2h6": 1, "c2h4": 2, "c2h2": 3, "co": 1}
    # Ngưỡng IEEE 90th percentile (có thể tính từ toàn bộ dữ liệu)
    ieee_90 = {"h2": 100, "ch4": 120, "c2h6": 65, "c2h4": 50, "c2h2": 35, "co": 350}
    for gas, w in gas_weights.items():
        if row[gas] > ieee_90.get(gas, 100):
            score += w * (row[gas] / ieee_90[gas])
    # Thêm điểm nếu có fault nghiêm trọng
    fault_risk = {"D2": 10, "D1": 7, "T3": 8, "T2": 5, "T1": 3, "PD": 2, "DT": 6}
    fault = row.get("consensus_fault", "INCONCLUSIVE")
    score += fault_risk.get(fault, 0)
    # Thêm nếu CO/CO2 cao
    if row["co"] > 500 or row["co2"] > 5000:
        score += 2
    return score

def apply_severity(df: pd.DataFrame) -> pd.DataFrame:
    df["severity_score"] = df.apply(calculate_severity, axis=1)
    # Phân loại mức
    conditions = [
        (df["severity_score"] < 5),
        (df["severity_score"] < 15),
        (df["severity_score"] < 30),
        (df["severity_score"] >= 30)
    ]
    choices = ["Normal", "Watchlist", "Warning", "Critical"]
    df["severity_level"] = np.select(conditions, choices, default="Critical")
    return df