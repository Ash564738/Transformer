from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List


@dataclass
class DiagnosticConfig:
    """
    DGA diagnostic configuration.

    Standard basis:
        IEEE Std C57.104-2019
        Mineral-oil-immersed transformers

    Important:
        The concentration limits below are the L1 limits used by
        the IEEE 2019 Doernenburg procedure.

        IEEE C57.104-2019 does NOT recommend using a fixed
        MIN_TDCG value such as 100 ppm as a universal "normal"
        criterion. Therefore TDCG is not used as the normal gate.
    """

    STANDARD: str = "IEEE C57.104-2019"
    FLUID_TYPE: str = "MINERAL_OIL"
    DGA_UNIT: str = "ppm"

    # --------------------------------------------------------
    # Fault grouping
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
        "MIXED": "MIXED",
        "ABSTAIN": "ABSTAIN",
    })

    # --------------------------------------------------------
    # Consensus
    #
    # Do NOT double-count Duval Pentagon P1 and P2.
    # Only one Pentagon should be included in this consensus.
    # --------------------------------------------------------

    METHOD_WEIGHTS: Dict[str, float] = field(default_factory=lambda: {
        "duval_pentagon_p2_fault": 1.0,
        "duval_triangle_fault": 1.0,
        "iec_fault": 1.0,
        "rogers_fault": 1.0,
        "doernenburg_fault": 1.0,
        "keygas_fault": 1.0,
    })

    MIXED_THRESHOLD: float = 0.65
    MIN_SECOND_GROUP_WEIGHT_RATIO: float = 0.30

    # Minimum number of independent diagnostic votes required
    # before calling a result a strong consensus.
    MIN_ACTIVE_METHODS_FOR_CONFIDENCE: int = 1

    # --------------------------------------------------------
    # IEEE C57.104-2019 L1 concentrations
    #
    # Doernenburg applicability limits
    # --------------------------------------------------------

    L1_LIMITS: Dict[str, float] = field(default_factory=lambda: {
        "h2": 100.0,
        "ch4": 120.0,
        "co": 350.0,
        "c2h2": 1.0,
        "c2h4": 50.0,
        "c2h6": 65.0,
    })

    # Explicit copy for Doernenburg.
    L1_DOERNENBURG: Dict[str, float] = field(default_factory=lambda: {
        "h2": 100.0,
        "ch4": 120.0,
        "co": 350.0,
        "c2h2": 1.0,
        "c2h4": 50.0,
        "c2h6": 65.0,
    })

    # --------------------------------------------------------
    # Rogers applicability
    #
    # Rogers is a ratio diagnostic method.
    # IEEE warns that it should not be used on samples with
    # very low gas levels.
    #
    # We therefore use the L1 limits as the conservative
    # applicability gate.
    # --------------------------------------------------------

    DIAGNOSTIC_GASES: List[str] = field(default_factory=lambda: [
        "h2",
        "ch4",
        "c2h2",
        "c2h4",
        "c2h6",
    ])

    # --------------------------------------------------------
    # Rogers ratio boundaries
    #
    # R1 = C2H2 / C2H4
    # R2 = CH4 / H2
    # R3 = C2H4 / C2H6
    #
    # IEEE C57.104-2019:
    #
    # R1:
    #   < 0.1       -> code 0
    #   0.1..3.0    -> code 1
    #   > 3.0       -> code 2
    #
    # R2:
    #   < 0.1       -> code 0
    #   0.1..1.0    -> code 1
    #   > 1.0       -> code 2
    #
    # R3:
    #   < 1.0       -> code 0
    #   1.0..3.0    -> code 1
    #   > 3.0       -> code 2
    # --------------------------------------------------------

    ROGERS_R1_LOW: float = 0.1
    ROGERS_R1_HIGH: float = 3.0

    ROGERS_R2_LOW: float = 0.1
    ROGERS_R2_HIGH: float = 1.0

    ROGERS_R3_LOW: float = 1.0
    ROGERS_R3_HIGH: float = 3.0

    # --------------------------------------------------------
    # Doernenburg ratio limits - mineral oil
    #
    # R1 = CH4 / H2
    # R2 = C2H2 / C2H4
    # R3 = C2H2 / CH4
    # R4 = C2H6 / C2H2
    #
    # IEEE C57.104-2019 Annex G
    # --------------------------------------------------------

    DOERNENBURG_OIL_LIMITS: Dict[str, Dict[str, float | str]] = field(
        default_factory=lambda: {
            "THERMAL": {
                "r1": "gt_1.0",
                "r2": "lt_0.75",
                "r3": "lt_0.3",
                "r4": "gt_0.4",
            },
            "PD": {
                "r1": "lt_0.1",
                "r2": "not_significant",
                "r3": "lt_0.3",
                "r4": "gt_0.4",
            },
            "D2": {
                "r1": "gt_0.1_lt_1.0",
                "r2": "gt_0.75",
                "r3": "gt_0.3",
                "r4": "lt_0.4",
            },
        }
    )

    # --------------------------------------------------------
    # Duval Triangle 1
    #
    # Coordinates are expressed as percentages:
    #
    # %CH4
    # %C2H4
    # %C2H2
    #
    # The actual classification polygons are implemented in
    # dga/duval_triangle.py.
    # --------------------------------------------------------

    DUVAL_TRIANGLE_GASES: List[str] = field(default_factory=lambda: [
        "ch4",
        "c2h4",
        "c2h2",
    ])

    DUVAL_MIN_TOTAL_GAS: float = 0.1

    # --------------------------------------------------------
    # Legacy / UI severity
    #
    # These values are NOT claimed to be the IEEE 2019
    # statistical normal limits.
    #
    # Keep only for backward compatibility with existing UI.
    # --------------------------------------------------------

    SEVERITY_LABELS: List[str] = field(default_factory=lambda: [
        "NORMAL",
        "WATCHLIST",
        "WARNING",
        "CRITICAL",
    ])

    SEVERITY_TO_UI: Dict[str, str] = field(default_factory=lambda: {
        "CRITICAL": "Severe",
        "WARNING": "Moderate",
        "WATCHLIST": "Low",
        "NORMAL": "Low",
    })

    SEVERITY_ACCENT: Dict[str, str] = field(default_factory=lambda: {
        "Severe": "red",
        "Moderate": "amber",
        "Low": "green",
    })

    # --------------------------------------------------------
    # Consensus policy
    # --------------------------------------------------------

    # Rogers NORMAL is a valid Rogers case, but NORMAL is not
    # treated as a fault vote.
    COUNT_NORMAL_AS_FAULT_VOTE: bool = False

    # DT is ambiguous by definition. It contributes to both
    # thermal and discharge evidence but is not allowed to
    # manufacture a specific fault label by itself.
    DT_IS_AMBIGUOUS: bool = True


config = DiagnosticConfig()


# ============================================================
# Paths
# ============================================================

BACKEND_ROOT = Path(__file__).resolve().parent

DATASET_DIR = BACKEND_ROOT / "dataset"
DATASET_DIR.mkdir(parents=True, exist_ok=True)

DATABASE_DIR = BACKEND_ROOT / "database"
DATABASE_DIR.mkdir(parents=True, exist_ok=True)

MODEL_DIR = BACKEND_ROOT / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)

REPORT_DIR = BACKEND_ROOT / "reports"
REPORT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Canonical fault labels
# ============================================================

FAULT_LABELS = [
    "NORMAL",
    "PD",
    "D1",
    "D2",
    "DT",
    "T1",
    "T2",
    "T3",
    "T3_H",
    "THERMAL_OIL",
    "THERMAL_CELLULOSE",
    "C",
    "O",
    "S",
    "ABSTAIN",
    "MIXED",
]

SEVERITY_LABELS = config.SEVERITY_LABELS