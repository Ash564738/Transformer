# config.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DiagnosticConfig:
    # --------------------------------------------------------
    # Fault grouping (logical mapping, không phải trọng số)
    # --------------------------------------------------------
    FAULT_GROUPS: Dict[str, str] = field(default_factory=lambda: {
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
        "C": "CELLULOSE",
        "O": "THERMAL",
        "S": "STRAY_GASSING",
        "ABSTAIN": "ABSTAIN",
        "MIXED": "MIXED"
    })

    # Method weights trong consensus – mặc định bằng nhau
    # (có thể điều chỉnh nếu có bằng chứng thực nghiệm)
    METHOD_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "duval_pentagon_p2_fault": 1.0,
        "duval_pentagon_p1_fault": 1.0,
        "duval_triangle_fault": 1.0,
        "iec_fault": 1.0,
        "rogers_fault": 1.0,
        "doernenburg_fault": 1.0,
        "keygas_fault": 1.0,
    })

    MIXED_THRESHOLD: float = 0.65
    MIN_SECOND_GROUP_WEIGHT_RATIO: float = 0.3

    # --------------------------------------------------------
    # Ngưỡng khí DGA – dựa trên IEEE C57.104 / IEC 60599
    # --------------------------------------------------------
    SEVERITY_GAS_THRESHOLDS: Dict[str, List[float]] = field(default_factory=lambda: {
        "h2": [100, 500, 1000],
        "ch4": [120, 400, 1000],
        "c2h6": [65, 100, 150],
        "c2h4": [50, 100, 200],
        "c2h2": [1, 9, 35],
        "co": [350, 700, 1400],
        "co2": [2500, 5000, 10000],
        "tcg": [720, 1920, 4630],
        "tdcg": [720, 1920, 4630],
    })

    # UI labels – không phải tham số khoa học, chỉ phục vụ hiển thị
    SEVERITY_LABELS: List[str] = field(default_factory=lambda: ["NORMAL", "WATCHLIST", "WARNING", "CRITICAL"])
    SEVERITY_TO_UI: Dict[str, str] = field(default_factory=lambda: {
        "CRITICAL": "Severe",
        "WARNING": "Moderate",
        "WATCHLIST": "Low",
        "NORMAL": "Low"
    })
    SEVERITY_ACCENT: Dict[str, str] = field(default_factory=lambda: {
        "Severe": "red", "Moderate": "amber", "Low": "green"
    })

    # DGA L1 limits – theo chuẩn
    L1_LIMITS: Dict[str, int] = field(default_factory=lambda: {
        "h2": 100, "ch4": 120, "c2h2": 1, "c2h4": 50, "c2h6": 65
    })
    L1_DOERNENBURG: Dict[str, int] = field(default_factory=lambda: {
        "h2": 100,
        "ch4": 120,
        "c2h2": 1,
        "c2h4": 50,
        "c2h6": 65,
    })
    MIN_TDCG: float = 100.0

config = DiagnosticConfig()

from pathlib import Path
BACKEND_ROOT = Path(__file__).resolve().parent
DATASET_DIR = BACKEND_ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)
DATABASE_DIR = BACKEND_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR = BACKEND_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)
REPORT_DIR = BACKEND_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)

FAULT_LABELS = [
    "NORMAL", "PD", "D1", "D2", "DT", "T1", "T2", "T3", "T3_H",
    "THERMAL_OIL", "THERMAL_CELLULOSE", "C", "O", "S", "ABSTAIN", "MIXED"
]
SEVERITY_LABELS = config.SEVERITY_LABELS