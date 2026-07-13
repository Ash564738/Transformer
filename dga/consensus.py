# consensus.py
import pandas as pd
import numpy as np

def consensus_vote(row):
    """Kết hợp các kết quả từ các phương pháp để đưa ra nhãn cuối cùng."""
    # Thu thập các dự đoán có sẵn
    votes = []
    if "iec_faults" in row:
        votes.extend(row["iec_faults"])  # list
    if "rogers_fault" in row and row["rogers_fault"] != "INCONCLUSIVE":
        votes.append(row["rogers_fault"])
    if "duval_triangle_fault" in row and row["duval_triangle_fault"] != "INCONCLUSIVE":
        votes.append(row["duval_triangle_fault"])
    # Thêm các phương pháp khác nếu có
    if not votes:
        return "INCONCLUSIVE"
    # Tìm nhãn phổ biến nhất
    label_counts = pd.Series(votes).value_counts()
    top_label = label_counts.index[0]
    if label_counts.iloc[0] >= 2:  # ít nhất 2 phiếu
        return top_label
    else:
        return "INCONCLUSIVE"

def apply_consensus(df: pd.DataFrame) -> pd.DataFrame:
    df["consensus_fault"] = df.apply(consensus_vote, axis=1)
    return df