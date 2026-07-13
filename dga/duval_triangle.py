# duval_triangle.py
import numpy as np
import pandas as pd

# Định nghĩa các vùng trong tam giác Duval 1 (theo tọa độ %CH4, %C2H4, %C2H2)
# Chỉ minh họa một vài vùng, thực tế cần tọa độ đầy đủ.
def duval_triangle_zone(pct_ch4, pct_c2h4, pct_c2h2):
    """Trả về mã vùng: PD, D1, D2, T1, T2, T3, DT."""
    # Chuyển đổi sang tọa độ tam giác (x, y) nếu cần, nhưng ở đây dùng logic đơn giản
    if pct_ch4 >= 98: return "PD"
    if pct_c2h2 >= 13 and pct_c2h4 < 23: return "D1"
    if pct_c2h2 >= 13 and pct_c2h4 >= 23 and pct_c2h2 < 29: return "D2"
    if pct_c2h4 >= 50 and pct_c2h2 < 4: return "T1" if pct_ch4 < 20 else "T2"
    if pct_c2h4 >= 50 and pct_c2h2 >= 4: return "T3"
    if 20 <= pct_c2h2 < 50 and pct_c2h4 >= 20: return "DT"
    return "INCONCLUSIVE"

def apply_duval_triangle(df: pd.DataFrame) -> pd.DataFrame:
    df["duval_triangle_fault"] = df.apply(
        lambda row: duval_triangle_zone(row["pct_ch4"], row["pct_c2h4"], row["pct_c2h2"]), axis=1)
    return df