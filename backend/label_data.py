# label_data.py
from pathlib import Path
import pandas as pd
import numpy as np
import consensus
import severity

DATA_DIR = Path("dataset/processed")
FEATURES_PATH = DATA_DIR / "dga_features.parquet"
LABELED_PATH = DATA_DIR / "dga_labeled.parquet"

def main():
    print("Loading features...")
    df = pd.read_parquet(FEATURES_PATH)

    # Chạy consensus (các cột khí cần có: h2, ch4, c2h6, c2h4, c2h2, co, co2)
    print("Applying consensus diagnosis...")
    df = consensus.apply_consensus(df)

    # Chạy severity
    print("Computing severity scores...")
    df = severity.apply_severity(df)

    # Đảm bảo có cột fault_type_label (map từ consensus_fault) để train_models.py dùng
    # Sử dụng danh sách nhãn thống nhất từ config (FAULT_LABELS)
    from config import FAULT_LABELS
    fault_map = {label: i for i, label in enumerate(FAULT_LABELS)}
    # Chuẩn hóa nhãn consensus về dạng trong FAULT_LABELS (nếu có khác biệt)
    # Hàm normalize_fault đã xử lý, nhưng ta cần đảm bảo tên khớp với FAULT_LABELS
    # Ở đây dùng chính cột consensus_fault, map sang index
    df["fault_type_label"] = df["consensus_fault"].map(fault_map).fillna(-1).astype(int)
    # Loại bỏ các dòng không có nhãn
    df = df[df["fault_type_label"] >= 0]

    # severity_label đã có từ apply_severity, ánh xạ sang index nếu cần
    sev_map = {"NORMAL": 0, "WATCHLIST": 1, "WARNING": 2, "CRITICAL": 3}
    df["severity_label"] = df["severity_label"].map(sev_map).fillna(0).astype(int)

    # severity_score đã có sẵn, giữ nguyên

    print(f"Saving labeled dataset to {LABELED_PATH}")
    df.to_parquet(LABELED_PATH, index=False)
    print(f"Done. Shape: {df.shape}")

if __name__ == "__main__":
    main()