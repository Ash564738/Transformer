# config.py
FAULT_LABELS = [
    "NORMAL",
    "PD",
    "D1",
    "D2",
    "T1",
    "T2",
    "T3",
    "THERMAL",
    "THERMAL_OIL",
    "THERMAL_CELLULOSE",
    "C",
    "O",
    "T3-H",
    "UNCERTAIN",
    "INVALID_LOW_GAS",
]

FAULT_GROUPS = {
    "NORMAL": "NORMAL",
    "PD": "DISCHARGE",
    "D1": "DISCHARGE",
    "D2": "DISCHARGE",
    "T1": "THERMAL",
    "T2": "THERMAL",
    "T3": "THERMAL",
    "MIXED": "MIXED",
    "CELLULOSE": "CELLULOSE",
    "UNCERTAIN": "UNCERTAIN",
    "INVALID_LOW_GAS": "INVALID_LOW_GAS",
}

SEVERITY_LABELS = [
    "Normal",
    "Watchlist",
    "Warning",
    "Critical",
]

SEVERITY_TO_UI = {
    "Critical": "Severe",
    "Warning": "Moderate",
    "Watchlist": "Low",
    "Normal": "Low",
}

SEVERITY_ACCENT = {
    "Severe": "red",
    "Moderate": "amber",
    "Low": "green",
}