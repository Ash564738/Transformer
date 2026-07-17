# config.py
FAULT_LABELS = [
    "NORMAL",
    "PD",                  # Partial Discharge (Phóng điện cục bộ)
    "D1",                  # Low energy electrical discharge (Phóng điện năng lượng thấp)
    "D2",                  # High energy electrical discharge / Arc (Phóng điện năng lượng cao)
    "DT",                  # Mixed electrical and thermal fault (Lỗi hỗn hợp)
    "T1",                  # Thermal fault < 300°C
    "T2",                  # Thermal fault 300–700°C
    "T3",                  # Thermal fault > 700°C
    "T3_H",                # Thermal fault > 700°C in oil only (Duval)
    "THERMAL_OIL",         # Overheating of oil only (IEEE Key Gas)
    "THERMAL_CELLULOSE",   # Overheating of cellulose (IEEE Key Gas)
    "C",                   # Carbonization of paper insulation (Duval Pentagon 2)
    "O",                   # Overheating < 250°C (Duval Pentagon 2)
    "S",                   # Stray Gassing / Khí đi lạc (Duval Pentagon 1)
    "UNCERTAIN",           # Lỗi không xác định / Tổ hợp tỷ số không hợp lệ
    "MIXED"
]

L1_DOERNENBURG = {
    "h2": 100,
    "ch4": 120,
    "c2h2": 35,
    "c2h4": 50,
    "c2h6": 65,
    # "co": 350,
    # "co2": 2500,
    # "tdcg": 720
}

# Ánh xạ về các nhóm lỗi chính để gom vote
FAULT_GROUPS = {
    "NORMAL": "NORMAL",
    "PD": "DISCHARGE",
    "D1": "DISCHARGE",
    "D2": "DISCHARGE",
    "DT": "MIXED",
    "T1": "THERMAL",
    "T2": "THERMAL",
    "T3": "THERMAL",
    "T3_H": "THERMAL",
    "THERMAL_OIL": "THERMAL",
    "THERMAL_CELLULOSE": "CELLULOSE",
    
    "C": "CELLULOSE", # Bản chất là giấy cách điện bị cháy than hóa
    "O": "THERMAL",               # Bản chất là quá nhiệt mức độ nhẹ < 250°C
    "S": "STRAY_GASSING",         # Hiện tượng sinh khí tự nhiên của dầu, không phải lỗi nguy hiểm
    
    "UNCERTAIN": "UNCERTAIN",
    "MIXED": "MIXED"
}

FAULT_EXPLANATIONS = {
    "NORMAL": "Normal Condition",
    "PD": "Partial Discharge",
    "D1": "Low energy electrical discharge (Sparking)",
    "D2": "High energy electrical discharge (Arcing)",
    "DT": "Mixed electrical and thermal fault",
    "T1": "Thermal fault < 300°C",
    "T2": "Thermal fault 300–700°C",
    "T3": "Thermal fault > 700°C",
    "T3_H": "Thermal fault > 700°C (oil only)",
    "THERMAL_OIL": "Thermal fault in oil only",
    "THERMAL_CELLULOSE": "Thermal fault involving cellulose insulation",
    "C": "Carbonization of paper insulation (Paper charring)",
    "O": "Overheating < 250°C (Mild thermal stress)",
    "S": "Stray gassing of oil",
    "UNCERTAIN": "Uncertain diagnosis",
    "MIXED": "Mixed fault"
}

FAULT_SEVERITY_POINTS = {
    "NORMAL": 0,
    "PD": 2,
    "D1": 3,
    "D2": 5,
    "DT": 5,
    "T1": 2,
    "T2": 3,
    "T3": 5,
    "T3_H": 5,
    "THERMAL_OIL": 3,
    "THERMAL_CELLULOSE": 4,
    "C": 5,
    "O": 2,
    "S": 1,
    "UNCERTAIN": 1,
    "MIXED": 5
}

# Trọng số cho từng phương pháp chẩn đoán
METHOD_WEIGHTS = {
    "duval_pentagon_p2_fault": 2.0,
    "duval_pentagon_p1_fault": 1.8,
    "duval_triangle_fault": 1.5,
    "iec_fault": 1.3,
    "rogers_fault": 1.1,
    "doernenburg_fault": 1.0,
    "keygas_fault": 0.8,
}

# Mức độ nghiêm trọng cho từng nhóm lỗi (dùng khi MIXED)
SEVERITY_BY_GROUP = {
    "NORMAL": 0,
    "DISCHARGE": 5,
    "THERMAL": 5,
    "CELLULOSE": 5,
    "STRAY_GASSING": 1,
    "MIXED": 5,        # fallback
    "UNCERTAIN": 1,
}

SEVERITY_LABELS = ["NORMAL", "WATCHLIST", "WARNING", "CRITICAL"]

SEVERITY_TO_UI = {
    "CRITICAL": "Severe",
    "WARNING": "Moderate",
    "WATCHLIST": "Low",
    "NORMAL": "Low",
}

SEVERITY_ACCENT = {
    "Severe": "red",
    "Moderate": "amber",
    "Low": "green",
}