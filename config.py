# config.py
FAULT_LABELS = ["Normal", "PD", "D1", "D2", "T1", "T2", "T3", "Cellulose", "Mixed", "Uncertain"]
SEVERITY_LABELS = ["Normal", "Watchlist", "Warning", "Critical", "Uncertain"]

# Mapping từ backend labels sang UI labels (Severe, Moderate, Low)
SEVERITY_TO_UI = {
    "Critical": "Severe",
    "Warning": "Moderate",
    "Watchlist": "Low",
    "Normal": "Low",
    "Uncertain": "Low",
}

# Màu accent hoặc class CSS cho từng UI label (tuỳ chọn)
SEVERITY_ACCENT = {
    "Severe": "red",
    "Moderate": "amber",
    "Low": "green",
}