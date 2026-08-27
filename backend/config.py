# config.py
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import ClassVar, Dict, Tuple


@dataclass
class DiagnosticConfig:
    """Auditable configuration for DGA inference and maintenance ranking.

    Design principles:
      * IEEE C57.104-2019 remains the primary DGA status engine.
      * No hand-assigned diagnostic or severity weights are used.
      * CRITICAL is an operational extreme-DGA extension, NOT IEEE Status 4.
      * Transformer ranking uses an unweighted lexicographic evidence order: current IEEE status dominates; history breaks ties.
      * Fault criticality is qualitative context only and never overrides IEEE status.
    """

    STANDARD: str = "IEEE C57.104-2019"
    IEC_STANDARD: str = "IEC 60599:2022"

    FLUID_TYPE: str = "MINERAL_OIL"
    DGA_UNIT: str = "ppm"

    BENCHMARK_FINE_CLASSES: ClassVar[Tuple[str, ...]] = (
        "NORMAL", "PD", "D1", "D2", "T1", "T2", "T3",
    )

    BENCHMARK_AMBIGUOUS_FINE_CLASSES: ClassVar[Tuple[str, ...]] = ("T1_T2",)
    BENCHMARK_AMBIGUOUS_ACCEPTED_PREDICTIONS: ClassVar[Dict[str, Tuple[str, ...]]] = {"T1_T2": ("T1", "T2")}

    COARSE_FAULT_GROUPS: ClassVar[Tuple[str, ...]] = (
        "NORMAL",
        "DISCHARGE",
        "THERMAL",
        "CELLULOSE",
        "STRAY_GASSING",
        "MIXED",
        "ABSTAIN",
    )

    FINE_FAULT_CLASSES: ClassVar[Tuple[str, ...]] = (
        "NORMAL", "PD", "D1", "D2", "T1", "T2", "T3",
        "T1_T2", "T3_H", "DT", "THERMAL_OIL", "THERMAL_CELLULOSE",
        "C", "O", "S", "MIXED", "ABSTAIN",
    )

    FAULT_GROUPS: ClassVar[Dict[str, str]] = {
        "NORMAL": "NORMAL",
        "PD": "DISCHARGE", "D1": "DISCHARGE", "D2": "DISCHARGE",
        "T1": "THERMAL", "T2": "THERMAL", "T3": "THERMAL",
        "T1_T2": "THERMAL", "T3_H": "THERMAL",
        "THERMAL_OIL": "THERMAL", "O": "THERMAL",
        "THERMAL_CELLULOSE": "CELLULOSE", "C": "CELLULOSE",
        "S": "STRAY_GASSING",
        "DT": "MIXED", "MIXED": "MIXED", "ABSTAIN": "ABSTAIN",
    }

    # Qualitative context only. This mapping is intentionally NOT a numeric
    # severity scale and is never included in maintenance ranking.
    # It is used to help engineers interpret the diagnosed fault type.
    FAULT_CRITICALITY_CONTEXT: ClassVar[Dict[str, str]] = {
        "NORMAL": "NO_FAULT",
        "PD": "LOWER_URGENCY",
        "D1": "ELEVATED_URGENCY",
        "D2": "HIGH_CONCERN",
        "T1": "LOWER_URGENCY",
        "T2": "ELEVATED_URGENCY",
        "T3": "HIGH_CONCERN",
        "T1_T2": "ELEVATED_URGENCY",
        "T3_H": "HIGH_CONCERN",
        "THERMAL_OIL": "ELEVATED_URGENCY",
        "THERMAL_CELLULOSE": "HIGH_CONCERN",
        "C": "HIGH_CONCERN",
        "O": "CONTEXT_DEPENDENT",
        "S": "CONTEXT_DEPENDENT",
        "DT": "CONTEXT_DEPENDENT",
        "MIXED": "CONTEXT_DEPENDENT",
        "ABSTAIN": "UNKNOWN",
    }

    FAULT_CRITICALITY_SOURCE: str = (
        "Qualitative fault-context layer informed by the standard fault taxonomy and "
        "transformer diagnostic literature; not a numeric severity weight and not used "
        "to override IEEE DGA status."
    )

    BENCHMARK_FAULT_ALIASES: ClassVar[Dict[str, str]] = {
        "NORMAL": "NORMAL",
        "PD": "PD", "PARTIAL_DISCHARGE": "PD", "PARTIAL DISCHARGE": "PD", "CORONA": "PD",
        "D1": "D1", "LOW_ENERGY_DISCHARGE": "D1", "LOW ENERGY DISCHARGE": "D1",
        "SPARK_DISCHARGE": "D1", "SPARK DISCHARGE": "D1",
        "D2": "D2", "HIGH_ENERGY_DISCHARGE": "D2", "HIGH ENERGY DISCHARGE": "D2",
        "ARC_DISCHARGE": "D2", "ARC DISCHARGE": "D2", "ARCING": "D2",
        "T1": "T1", "LOW_TEMPERATURE_THERMAL": "T1", "LOW TEMPERATURE THERMAL": "T1",
        "LOW_TEMPERATURE_OVERHEATING": "T1", "LOW TEMPERATURE OVERHEATING": "T1",
        "T2": "T2", "MEDIUM_TEMPERATURE_THERMAL": "T2", "MEDIUM TEMPERATURE THERMAL": "T2",
        "MIDDLE_TEMPERATURE_OVERHEATING": "T2", "MIDDLE TEMPERATURE OVERHEATING": "T2",
        "T3": "T3", "HIGH_TEMPERATURE_THERMAL": "T3", "HIGH TEMPERATURE THERMAL": "T3",
        "HIGH_TEMPERATURE_OVERHEATING": "T3", "HIGH TEMPERATURE OVERHEATING": "T3",
        "T1_T2": "T1_T2", "T1/T2": "T1_T2",
        "LOW_MIDDLE_TEMPERATURE_OVERHEATING": "T1_T2",
        "LOW/MIDDLE-TEMPERATURE OVERHEATING": "T1_T2",
        "LOW/MIDDLE TEMPERATURE OVERHEATING": "T1_T2",
        "LOW MIDDLE TEMPERATURE OVERHEATING": "T1_T2",
        "THERMAL_OIL": "THERMAL_OIL", "O": "O",
        "THERMAL_CELLULOSE": "THERMAL_CELLULOSE",
        "CELLULOSE": "C", "C": "C",
        "STRAY_GASSING": "S", "S": "S",
        "DT": "DT", "MIXED": "MIXED", "ABSTAIN": "ABSTAIN",
    }

    DIAGNOSTIC_METHODS: ClassVar[Tuple[str, ...]] = (
        "keygas_fault", "iec_fault", "rogers_fault",
        "doernenburg_fault", "duval_triangle_fault",
        "duval_pentagon_p1_fault", "duval_pentagon_p2_fault",
    )
    DIAGNOSTIC_METHOD_TO_COLUMN: ClassVar[Dict[str, str]] = {
        m: m for m in DIAGNOSTIC_METHODS
    }

    COMMON_BENCHMARK_GASES: ClassVar[Tuple[str, ...]] = (
        "h2", "ch4", "c2h6", "c2h4", "c2h2",
    )
    FULL_EXTERNAL_GASES: ClassVar[Tuple[str, ...]] = (
        "h2", "ch4", "c2h6", "c2h4", "c2h2", "co", "co2",
    )
    ALL_DGA_GASES: ClassVar[Tuple[str, ...]] = FULL_EXTERNAL_GASES
    SEVERITY_REQUIRED_GASES: ClassVar[Tuple[str, ...]] = FULL_EXTERNAL_GASES

    TABLE_1_90TH: ClassVar[Dict[str, Dict[str, Dict[str, float]]]] = {
        "LE_0_2": {
            "unknown": {"h2": 80, "ch4": 90, "c2h6": 90, "c2h4": 50, "c2h2": 1, "co": 900, "co2": 9000},
            "1_9": {"h2": 75, "ch4": 45, "c2h6": 30, "c2h4": 20, "c2h2": 1, "co": 900, "co2": 5000},
            "10_30": {"h2": 90, "ch4": 90, "c2h6": 90, "c2h4": 50, "c2h2": 1, "co": 900, "co2": 10000},
            "gt_30": {"h2": 100, "ch4": 110, "c2h6": 150, "c2h4": 90, "c2h2": 1, "co": 900, "co2": 10000},
        },
        "GT_0_2": {
            "unknown": {"h2": 40, "ch4": 20, "c2h6": 15, "c2h4": 50, "c2h2": 2, "co": 500, "co2": 5000},
            "1_9": {"h2": 40, "ch4": 20, "c2h6": 15, "c2h4": 25, "c2h2": 2, "co": 500, "co2": 3500},
            "10_30": {"h2": 40, "ch4": 20, "c2h6": 15, "c2h4": 60, "c2h2": 2, "co": 500, "co2": 5500},
            "gt_30": {"h2": 40, "ch4": 20, "c2h6": 15, "c2h4": 60, "c2h2": 2, "co": 500, "co2": 5500},
        },
    }

    TABLE_2_95TH: ClassVar[Dict[str, Dict[str, Dict[str, float]]]] = {
        "LE_0_2": {
            "unknown": {"h2": 200, "ch4": 150, "c2h6": 175, "c2h4": 100, "c2h2": 2, "co": 1100, "co2": 12500},
            "1_9": {"h2": 200, "ch4": 100, "c2h6": 70, "c2h4": 40, "c2h2": 2, "co": 1100, "co2": 7000},
            "10_30": {"h2": 200, "ch4": 150, "c2h6": 175, "c2h4": 95, "c2h2": 2, "co": 1100, "co2": 14000},
            "gt_30": {"h2": 200, "ch4": 250, "c2h6": 175, "c2h4": 175, "c2h2": 4, "co": 1100, "co2": 14000},
        },
        "GT_0_2": {
            "unknown": {"h2": 90, "ch4": 50, "c2h6": 40, "c2h4": 100, "c2h2": 7, "co": 600, "co2": 7000},
            "1_9": {"h2": 90, "ch4": 60, "c2h6": 30, "c2h4": 80, "c2h2": 7, "co": 600, "co2": 5000},
            "10_30": {"h2": 90, "ch4": 60, "c2h6": 40, "c2h4": 125, "c2h2": 7, "co": 600, "co2": 8000},
            "gt_30": {"h2": 90, "ch4": 30, "c2h6": 40, "c2h4": 125, "c2h2": 7, "co": 600, "co2": 8000},
        },
    }

    TABLE_3_DELTA_95TH: ClassVar[Dict[str, Dict[str, float | None]]] = {
        "LE_0_2": {"h2": 40, "ch4": 30, "c2h6": 25, "c2h4": 20, "c2h2": None, "co": 250, "co2": 2500},
        "GT_0_2": {"h2": 25, "ch4": 10, "c2h6": 7, "c2h4": 20, "c2h2": None, "co": 175, "co2": 1750},
    }

    TABLE_4_RATE_95TH: ClassVar[Dict[str, Dict[str, Dict[str, float | None]]]] = {
        "LE_0_2": {
            "4_9": {"h2": 25, "ch4": 4, "c2h6": 3, "c2h4": 7, "c2h2": None, "co": 100, "co2": 1000},
            "10_24": {"h2": 10, "ch4": 3, "c2h6": 2, "c2h4": 5, "c2h2": None, "co": 80, "co2": 800},
        },
        "GT_0_2": {
            "4_9": {"h2": 50, "ch4": 15, "c2h6": 15, "c2h4": 10, "c2h2": None, "co": 200, "co2": 1750},
            "10_24": {"h2": 20, "ch4": 10, "c2h6": 9, "c2h4": 7, "c2h2": None, "co": 100, "co2": 1000},
        },
    }

    SEVERITY_LABELS: ClassVar[Tuple[str, ...]] = (
        "INSUFFICIENT_DATA", "STATUS_1", "STATUS_2", "STATUS_3"
    )
    SEVERITY_ORDER: ClassVar[Dict[str, int]] = {
        "INSUFFICIENT_DATA": 0, "STATUS_1": 1, "STATUS_2": 2, "STATUS_3": 3,
    }
    ORDINAL_TO_SEVERITY: ClassVar[Dict[int, str]] = {v: k for k, v in SEVERITY_ORDER.items()}

    RATE_MIN_POINTS: int = 3
    RATE_MAX_POINTS: int = 6
    RATE_MIN_MONTHS: float = 4.0
    RATE_MAX_MONTHS: float = 24.0
    MIN_RECORDS_FOR_OBSERVED_TREND: int = 3

    USE_WEIGHTED_SEVERITY_SCORE: bool = False
    USE_FAILURE_PROBABILITY_AS_SEVERITY: bool = False

    # Deprecated compatibility fields. No additional Status-4/Critical class is used.
    CRITICAL_RULE: str = "NOT_USED"
    CRITICAL_REFERENCE: str = (
        "No additional severity class is created. IEEE Status 1/2/3 remain the condition classes; "
        "fleet priority uses an ordinal current-status score plus bounded historical evidence."
    )

    MAINTENANCE_PRIORITY_LABELS: ClassVar[Dict[int, str]] = {
        3: "HIGH_RISK",
        2: "WATCH",
        1: "NORMAL",
        0: "DATA_REVIEW",
    }

    RANKING_POLICY: ClassVar[Tuple[str, ...]] = (
        "current IEEE DGA status (Status 3 > Status 2 > Status 1)",
        "current concentration exceedance ratio against the applicable IEEE concentration threshold",
        "current rate exceedance ratio against the applicable IEEE rate threshold (when available)",
        "current delta exceedance ratio against the applicable IEEE delta threshold (when available)",
        "number of independent current IEEE trigger tables",
        "current per-table exceedance counts",
        "historical maximum IEEE status before the current sample",
        "historical maximum concentration exceedance before the current sample",
    )
    RANKING_TIE_POLICY: str = (
        "Identical evidence vectors receive the same rank; missing history never lowers current priority. No hand-assigned numeric weights are used."
    )

    WEAK_ABSTAIN_LABEL: int = -1
    WEAK_COARSE_GROUPS: ClassVar[Tuple[str, ...]] = (
        "NORMAL", "DISCHARGE", "THERMAL", "CELLULOSE", "STRAY_GASSING", "MIXED",
    )
    WEAK_FINE_GROUPS: ClassVar[Tuple[str, ...]] = BENCHMARK_FINE_CLASSES

    CLASSICAL_ML_MODELS: ClassVar[Tuple[str, ...]] = (
        "logistic_regression", "random_forest", "extra_trees",
        "svm_rbf", "hist_gradient_boosting", "knn", "sklearn_mlp",
    )
    # Operational weak-student training is intentionally smaller than the
    # external benchmark grid. The latter remains broad for research comparison.
    WEAK_STUDENT_MODELS: ClassVar[Tuple[str, ...]] = (
        "logistic_regression",
        "random_forest",
        "extra_trees",
        "hist_gradient_boosting",
        "sklearn_mlp",
    )
    STUDENT_FEATURE_MODES: ClassVar[Tuple[str, ...]] = (
        "gas_only", "gas_plus_traditional",
    )
    PRIMARY_SELECTION_METRIC: str = "macro_f1"
    HYBRID_METHODS: ClassVar[Tuple[str, ...]] = (
        "student_only", "agreement_only",
    )
    RANDOM_STATE: int = 42
    DEV_SIZE: float = 0.20
    TEST_SIZE: float = 0.20
    BOOTSTRAP_ITERATIONS: int = 2000
    BOOTSTRAP_CONFIDENCE: float = 0.95

    FAULT_METRICS: ClassVar[Tuple[str, ...]] = (
        "accuracy", "balanced_accuracy", "macro_precision", "macro_recall",
        "macro_f1", "weighted_f1", "coverage", "selective_accuracy",
    )
    SEVERITY_METRICS: ClassVar[Tuple[str, ...]] = (
        "accuracy", "balanced_accuracy", "macro_f1", "ordinal_mae",
        "quadratic_weighted_kappa",
    )

    UNLABELED_RAW_COLUMNS: ClassVar[Tuple[str, ...]] = (
        "loc", "name", "codetx", "mfg", "ser", "kv", "mva", "year_energized",
        "sample_day", "tested_day", "o2", "n2", "co2", "co", "h2", "ch4",
        "c2h2", "c2h4", "c2h6", "c3h6", "c3h8", "tcg", "temp", "water", "nb",
    )

    BACKEND_ROOT: Path = field(default_factory=lambda: Path(__file__).resolve().parent, init=False, repr=False)

    @property
    def DATASET_DIR(self) -> Path:
        return self.BACKEND_ROOT / "dataset"

    @property
    def DATABASE_DIR(self) -> Path:
        return self.BACKEND_ROOT / "database"

    @property
    def MODEL_DIR(self) -> Path:
        return self.BACKEND_ROOT / "models"

    @property
    def REPORT_DIR(self) -> Path:
        return self.BACKEND_ROOT / "reports"

    def ensure_directories(self) -> None:
        for directory in (
            self.DATASET_DIR,
            self.DATABASE_DIR,
            self.MODEL_DIR,
            self.REPORT_DIR,
            self.DATASET_DIR / "processed",
            self.REPORT_DIR / "benchmark",
        ):
            directory.mkdir(parents=True, exist_ok=True)


config = DiagnosticConfig()
config.ensure_directories()

BACKEND_ROOT = config.BACKEND_ROOT
DATASET_DIR = config.DATASET_DIR
DATABASE_DIR = config.DATABASE_DIR
MODEL_DIR = config.MODEL_DIR
REPORT_DIR = config.REPORT_DIR

FAULT_LABELS = list(config.FINE_FAULT_CLASSES)
SEVERITY_LABELS = list(config.SEVERITY_LABELS)
COARSE_FAULT_GROUPS = list(config.COARSE_FAULT_GROUPS)
WEAK_GROUPS = list(config.WEAK_COARSE_GROUPS)
DIAGNOSTIC_METHODS = list(config.DIAGNOSTIC_METHODS)
CORE_GASES = list(config.FULL_EXTERNAL_GASES)
COMBUSTIBLE_GASES = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "co"]
OPTIONAL_GASES = ["o2", "n2", "c3h6", "c3h8"]
SEVERITY_ORDINAL = dict(config.SEVERITY_ORDER)