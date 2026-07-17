# config.py
from dataclasses import dataclass, field
from typing import Dict, List

@dataclass
class DiagnosticConfig:
    # Fault grouping
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
        "UNCERTAIN": "UNCERTAIN",
        "MIXED": "MIXED"
    })

    # Method weights for consensus
    METHOD_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "duval_pentagon_p2_fault": 2.0,
        "duval_pentagon_p1_fault": 1.8,
        "duval_triangle_fault": 1.5,
        "iec_fault": 1.3,
        "rogers_fault": 1.1,
        "doernenburg_fault": 1.0,
        "keygas_fault": 1.2,
    })

    MIXED_THRESHOLD: float = 0.65
    MIN_SECOND_GROUP_WEIGHT_RATIO: float = 0.3

    # Severity thresholds per gas [Level1, Level2, Level3]
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

    SEVERITY_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "gas": 1.0,
        "trend": 1.2,
        "fault": 1.0,
        "aging": 0.8
    })

    SEVERITY_CLASS_BOUNDARIES: List[float] = field(default_factory=lambda: [4, 8, 13])

    FAULT_SEVERITY_POINTS: Dict[str, int] = field(default_factory=lambda: {
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
    })

    SEVERITY_BY_GROUP: Dict[str, int] = field(default_factory=lambda: {
        "NORMAL": 0,
        "DISCHARGE": 5,
        "THERMAL": 5,
        "CELLULOSE": 5,
        "STRAY_GASSING": 1,
        "MIXED": 5,
        "UNCERTAIN": 1,
    })

    # Ranking
    RANKING_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "current": 0.55,
        "history": 0.25,
        "trend": 0.10,
        "confidence": 0.05,
        "critical_history": 0.05
    })
    PERSISTENCE_BONUS_FACTOR: float = 0.15
    TREND_SLOPE_WORSENING: float = 0.5
    TREND_SLOPE_IMPROVING: float = -0.5
    RECENT_SAMPLES_FOR_TREND: int = 5
    RECENT_SAMPLES_FOR_HISTORY: int = 5
    RECENT_SAMPLES_FOR_PERSISTENCE: int = 5
    ACTION_THRESHOLD_CRITICAL: float = 12.0
    ACTION_THRESHOLD_WARNING: float = 6.0

    # UI
    SEVERITY_LABELS: List[str] = field(default_factory=lambda: ["NORMAL", "WATCHLIST", "WARNING", "CRITICAL"])
    SEVERITY_TO_UI: Dict[str, str] = field(default_factory=lambda: {
        "CRITICAL": "Severe",
        "WARNING": "Moderate",
        "WATCHLIST": "Low",
        "NORMAL": "Low"
    })
    SEVERITY_ACCENT: Dict[str, str] = field(default_factory=lambda: {
        "Severe": "red",
        "Moderate": "amber",
        "Low": "green"
    })

    # DGA L1 limits
    L1_LIMITS: Dict[str, int] = field(default_factory=lambda: {  # IEC, Rogers, Keygas
        "h2": 100,
        "ch4": 120,
        "c2h2": 1,
        "c2h4": 50,
        "c2h6": 65
    })
    L1_DOERNENBURG: Dict[str, int] = field(default_factory=lambda: {
        "h2": 100,
        "ch4": 120,
        "c2h2": 35,
        "c2h4": 50,
        "c2h6": 65
    })
    MIN_TDCG: float = 100.0

config = DiagnosticConfig()