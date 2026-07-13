# keygas.py
import pandas as pd
import numpy as np

def key_gas_method(row):
    """Xác định fault dựa trên khí chiếm ưu thế."""
    gases = {"h2": "PD", "ch4": "PD", "c2h6": "T1", "c2h4": "T3", "c2h2": "D2", "co": "CELLULOSE"}
    # Lấy khí có nồng độ cao nhất
    max_gas = max(gases.keys(), key=lambda g: row.get(g, 0))
    return gases[max_gas]

def apply_key_gas(df: pd.DataFrame) -> pd.DataFrame:
    df["keygas_fault"] = df.apply(key_gas_method, axis=1)
    return df