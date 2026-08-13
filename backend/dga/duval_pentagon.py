# dga/duval_pentagon.py

from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Duval Pentagon coordinate system
# ============================================================================
#
# Standard Duval Pentagon coordinate system:
#
#                  H2 (0, 40)
#                    /\
#                   /  \
#                  /    \
#        C2H6 ----/      \---- C2H2
#        (-38,12.4)      (38,12.4)
#                 \      /
#                  \    /
#                   \  /
#          CH4 -----\/----- C2H4
#        (-23.5,-32.4)  (23.5,-32.4)
#
# Gas order around the pentagon:
# H2 -> C2H6 -> CH4 -> C2H4 -> C2H2
#
# IMPORTANT:
# Do NOT apply an arbitrary SCALE factor to the standard coordinates.
# The published zone boundaries are already expressed in this coordinate
# system.
# ============================================================================


PENTAGON_VERTICES_RAW: Dict[str, np.ndarray] = {
    "H2": np.array([0.0, 40.0], dtype=float),
    "C2H6": np.array([-38.0, 12.4], dtype=float),
    "CH4": np.array([-23.5, -32.4], dtype=float),
    "C2H4": np.array([23.5, -32.4], dtype=float),
    "C2H2": np.array([38.0, 12.4], dtype=float),
}

# Public alias retained for compatibility with existing code.
PENTAGON_VERTICES = PENTAGON_VERTICES_RAW


# ============================================================================
# Zone definitions
# ============================================================================
#
# Coordinates are the standard Duval Pentagon coordinates.
#
# Pentagon 1:
#   PD, D1, D2, T1, T2, T3, S
#
# Pentagon 2:
#   PD, D1, D2, T3-H, C, O, S
#
# The numerical zone coordinates follow the published Duval/IEEE geometry.
# ============================================================================


def _zone(points: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    """Convert zone coordinates to immutable float tuples."""
    return tuple((float(x), float(y)) for x, y in points)


PENTAGON_1_ZONES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "PD": _zone(
        [
            (0.0, 33.0),
            (-1.0, 33.0),
            (-1.0, 24.5),
            (0.0, 24.5),
        ]
    ),

    "D1": _zone(
        [
            (0.0, 40.0),
            (38.0, 12.0),
            (32.0, -6.1),
            (4.0, 16.0),
            (0.0, 1.5),
        ]
    ),

    "D2": _zone(
        [
            (4.0, 16.0),
            (32.0, -6.1),
            (24.3, -30.0),
            (0.0, -3.0),
            (0.0, 1.5),
        ]
    ),

    "T3": _zone(
        [
            (0.0, -3.0),
            (24.3, -30.0),
            (23.5, -32.4),
            (1.0, -32.4),
            (-6.0, -4.0),
        ]
    ),

    "T2": _zone(
        [
            (-6.0, -4.0),
            (1.0, -32.4),
            (-22.5, -32.4),
        ]
    ),

    "T1": _zone(
        [
            (-6.0, -4.0),
            (-22.5, -32.4),
            (-23.5, -32.4),
            (-35.0, 3.0),
            (0.0, 1.5),
            (0.0, -3.0),
        ]
    ),

    "S": _zone(
        [
            (0.0, 1.5),
            (-35.0, 3.1),
            (-38.0, 12.4),
            (0.0, 40.0),
            (0.0, 33.0),
            (-1.0, 33.0),
            (-1.0, 24.5),
            (0.0, 24.5),
        ]
    ),
}


PENTAGON_2_ZONES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    # Common zones shared with Pentagon 1.
    "PD": PENTAGON_1_ZONES["PD"],
    "D1": PENTAGON_1_ZONES["D1"],
    "D2": PENTAGON_1_ZONES["D2"],
    "S": PENTAGON_1_ZONES["S"],

    "T3_H": _zone(
        [
            (0.0, -3.0),
            (24.3, -30.0),
            (23.5, -32.4),
            (2.5, -32.4),
            (-3.5, -3.0),
        ]
    ),

    "C": _zone(
        [
            (-3.5, -3.0),
            (2.5, -32.4),
            (-21.5, -32.4),
            (-11.0, -8.0),
        ]
    ),

    "O": _zone(
        [
            (-3.5, -3.0),
            (-11.0, -8.0),
            (-21.5, -32.4),
            (-23.5, -32.4),
            (-35.0, 3.1),
            (0.0, 1.5),
            (0.0, -3.0),
        ]
    ),
}


# ============================================================================
# Display information
# ============================================================================

ZONE_COLORS: Dict[str, str] = {
    "PD": "#cfff7c",
    "D1": "#ffffb3",
    "D2": "#cec9ec",
    "T1": "#9dd6ff",
    "T2": "#ffbd72",
    "T3": "#9de4d9",
    "T3_H": "#90ccc2",
    "C": "#fb8072",
    "O": "#ffd3ea",
    "S": "#ffcfab",
}


FAULT_EXPLANATIONS: Dict[str, str] = {
    "PD": "Partial Discharge",
    "D1": "Low energy electrical discharge",
    "D2": "High energy electrical discharge (arc)",
    "T1": "Thermal fault < 300°C",
    "T2": "Thermal fault 300–700°C",
    "T3": "Thermal fault > 700°C",
    "T3_H": "Thermal fault > 700°C (oil only)",
    "C": "Carbonization of paper insulation",
    "O": "Overheating < 250°C",
    "S": "Stray Gassing",
}


# ============================================================================
# Pre-built polygon paths
# ============================================================================

PATHS_P1: Dict[str, Path] = {
    name: Path(poly)
    for name, poly in PENTAGON_1_ZONES.items()
}

PATHS_P2: Dict[str, Path] = {
    name: Path(poly)
    for name, poly in PENTAGON_2_ZONES.items()
}


# ============================================================================
# Input validation
# ============================================================================

PENTAGON_GASES = (
    "h2",
    "ch4",
    "c2h6",
    "c2h4",
    "c2h2",
)

# Minimum total concentration required before attempting a Pentagon diagnosis.
#
# This is only a numerical validity gate. It does NOT mean that Duval
# Pentagon itself defines "0.1 ppm" as a diagnostic threshold.
#
# Actual applicability / fault interpretation should remain consistent with
# the overall DGA pipeline.
MIN_PENTAGON_TOTAL = 0.1


def _safe_gas(value) -> float:
    """
    Safely convert a gas concentration to a finite non-negative float.

    Invalid, missing, infinite, or negative values return NaN so that the
    caller can distinguish invalid data from a legitimate zero concentration.
    """
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    if value < 0:
        return np.nan

    return value


def _read_pentagon_gases(row: pd.Series) -> Optional[Tuple[float, ...]]:
    """
    Read and validate the five gases required by Duval Pentagon.

    Returns:
        Tuple(h2, ch4, c2h6, c2h4, c2h2)
        or None when the input is invalid.
    """
    values = tuple(
        _safe_gas(row.get(gas, np.nan))
        for gas in PENTAGON_GASES
    )

    if any(np.isnan(value) for value in values):
        return None

    return values


# ============================================================================
# Polygon centroid
# ============================================================================

def _polygon_centroid(
    points: Iterable[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:
    """
    Calculate the geometric centroid of a polygon.

    Uses the standard shoelace / area-centroid formula.

    Returns:
        (x, y) centroid, or None for a degenerate polygon.
    """
    pts = np.asarray(list(points), dtype=float)

    if pts.ndim != 2 or pts.shape[0] < 3 or pts.shape[1] != 2:
        return None

    if not np.all(np.isfinite(pts)):
        return None

    x = pts[:, 0]
    y = pts[:, 1]

    # Close polygon.
    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = x * y_next - x_next * y

    signed_area = 0.5 * np.sum(cross)

    if not np.isfinite(signed_area):
        return None

    # Degenerate polygon.
    if abs(signed_area) <= 1e-12:
        return None

    cx = (
        np.sum((x + x_next) * cross)
        / (6.0 * signed_area)
    )

    cy = (
        np.sum((y + y_next) * cross)
        / (6.0 * signed_area)
    )

    if not np.isfinite(cx) or not np.isfinite(cy):
        return None

    return float(cx), float(cy)


# ============================================================================
# Duval Pentagon centroid
# ============================================================================

def duval_pentagon_centroid(
    h2: float,
    ch4: float,
    c2h6: float,
    c2h4: float,
    c2h2: float,
) -> Optional[Tuple[float, float]]:
    """
    Calculate the Duval Pentagon centroid.

    The five gas concentrations are first normalized to percentages.

    Five points are then constructed by scaling each pentagon vertex by
    the corresponding normalized gas fraction:

        P_H2   = fraction(H2)   * V_H2
        P_C2H6  = fraction(C2H6) * V_C2H6
        P_CH4   = fraction(CH4)  * V_CH4
        P_C2H4  = fraction(C2H4) * V_C2H4
        P_C2H2  = fraction(C2H2) * V_C2H2

    The centroid of that resulting polygon is then calculated.

    This is intentionally NOT a simple weighted average of the five
    pentagon vertices.
    """

    values = (
        _safe_gas(h2),
        _safe_gas(ch4),
        _safe_gas(c2h6),
        _safe_gas(c2h4),
        _safe_gas(c2h2),
    )

    if any(np.isnan(value) for value in values):
        return None

    h2, ch4, c2h6, c2h4, c2h2 = values

    total = h2 + ch4 + c2h6 + c2h4 + c2h2

    if not np.isfinite(total) or total < MIN_PENTAGON_TOTAL:
        return None

    fractions = {
        "H2": h2 / total,
        "C2H6": c2h6 / total,
        "CH4": ch4 / total,
        "C2H4": c2h4 / total,
        "C2H2": c2h2 / total,
    }

    # IMPORTANT:
    # This is the gas-order geometry used by the Duval Pentagon:
    #
    # H2 -> C2H6 -> CH4 -> C2H4 -> C2H2
    #
    # Do not reorder this sequence.
    polygon_points = [
        tuple(
            fractions["H2"] * PENTAGON_VERTICES["H2"]
        ),
        tuple(
            fractions["C2H6"] * PENTAGON_VERTICES["C2H6"]
        ),
        tuple(
            fractions["CH4"] * PENTAGON_VERTICES["CH4"]
        ),
        tuple(
            fractions["C2H4"] * PENTAGON_VERTICES["C2H4"]
        ),
        tuple(
            fractions["C2H2"] * PENTAGON_VERTICES["C2H2"]
        ),
    ]

    return _polygon_centroid(polygon_points)


# ============================================================================
# Zone lookup
# ============================================================================

def _find_pentagon_zone(
    xy: Optional[Tuple[float, float]],
    paths: Dict[str, Path],
) -> str:
    """
    Find the Duval Pentagon zone containing the centroid.

    Returns:
        Fault label or ABSTAIN.
    """
    if xy is None:
        return "ABSTAIN"

    x, y = xy

    if not np.isfinite(x) or not np.isfinite(y):
        return "ABSTAIN"

    point = (float(x), float(y))

    for zone, path in paths.items():
        # A very small positive radius makes points exactly on a zone
        # boundary less likely to be rejected because of floating-point
        # roundoff. It does not materially expand the diagnostic zones.
        if path.contains_point(point, radius=1e-9):
            return zone

    return "ABSTAIN"


# ============================================================================
# Single-row classification
# ============================================================================

def classify_duval_pentagon(
    row: pd.Series,
) -> Tuple[Optional[Tuple[float, float]], str, str]:
    """
    Classify one DGA sample using both Duval Pentagon 1 and 2.

    Returns:
        (centroid, p1_fault, p2_fault)
    """

    gases = _read_pentagon_gases(row)

    # Invalid data must never become NORMAL.
    if gases is None:
        return None, "ABSTAIN", "ABSTAIN"

    h2, ch4, c2h6, c2h4, c2h2 = gases

    total = h2 + ch4 + c2h6 + c2h4 + c2h2

    # Very low total gas is not a confirmed NORMAL diagnosis.
    # The method simply has insufficient usable signal.
    if not np.isfinite(total) or total < MIN_PENTAGON_TOTAL:
        return None, "ABSTAIN", "ABSTAIN"

    xy = duval_pentagon_centroid(
        h2=h2,
        ch4=ch4,
        c2h6=c2h6,
        c2h4=c2h4,
        c2h2=c2h2,
    )

    if xy is None:
        return None, "ABSTAIN", "ABSTAIN"

    p1_fault = _find_pentagon_zone(
        xy,
        PATHS_P1,
    )

    p2_fault = _find_pentagon_zone(
        xy,
        PATHS_P2,
    )

    return xy, p1_fault, p2_fault


# ============================================================================
# Apply both Pentagon 1 and Pentagon 2
# ============================================================================

def apply_duval_pentagon_dual(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply Duval Pentagon 1 and Pentagon 2 simultaneously.

    Output columns:

        p_x
        p_y
        fault_p1
        fault_p2
        duval_pentagon_p1_fault
        duval_pentagon_p2_fault

    IMPORTANT:
    Pentagon 1 and Pentagon 2 are two views of the SAME gas sample.
    They should NOT be treated as two statistically independent voting
    methods in consensus logic.
    """

    df = df.copy()

    xs = []
    ys = []
    faults_p1 = []
    faults_p2 = []

    for _, row in df.iterrows():
        xy, fault_p1, fault_p2 = classify_duval_pentagon(row)

        if xy is None:
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))

        faults_p1.append(fault_p1)
        faults_p2.append(fault_p2)

    df["p_x"] = xs
    df["p_y"] = ys

    # Internal / backward-compatible names.
    df["fault_p1"] = faults_p1
    df["fault_p2"] = faults_p2

    # Explicit method names.
    #
    # These are both exported, but they represent ONE Duval Pentagon
    # diagnostic family, not two independent voting sources.
    df["duval_pentagon_p1_fault"] = faults_p1
    df["duval_pentagon_p2_fault"] = faults_p2

    logger.debug(
        "Duval Pentagon P1/P2 faults applied."
    )

    if logger.isEnabledFor(logging.DEBUG):
        debug_cols = [
            "h2",
            "ch4",
            "c2h6",
            "c2h4",
            "c2h2",
            "p_x",
            "p_y",
            "duval_pentagon_p1_fault",
            "duval_pentagon_p2_fault",
        ]

        available_cols = [
            col for col in debug_cols
            if col in df.columns
        ]

        logger.debug(
            "Sample Duval Pentagon results:\n%s",
            df[available_cols].head(5).to_string(),
        )

    return df


# ============================================================================
# Backward-compatible single Pentagon API
# ============================================================================

def apply_duval_pentagon(
    df: pd.DataFrame,
    pentagon: str = "P2",
) -> pd.DataFrame:
    """
    Backward-compatible wrapper.

    The function always calculates BOTH P1 and P2.

    `pentagon` only determines which result is copied into the legacy
    `duval_pentagon_fault` column.

    Args:
        df: Input DGA DataFrame.
        pentagon: "P1" or "P2". Default is "P2".

    Returns:
        DataFrame containing both Pentagon results.
    """

    df = apply_duval_pentagon_dual(df)

    pentagon_name = str(pentagon).upper().strip()

    if pentagon_name == "P1":
        df["duval_pentagon_fault"] = (
            df["duval_pentagon_p1_fault"]
        )

    else:
        # Default and backward-compatible behavior.
        df["duval_pentagon_fault"] = (
            df["duval_pentagon_p2_fault"]
        )

    return df


# ============================================================================
# Optional helper: explicit P1/P2 result access
# ============================================================================

def get_duval_pentagon_faults(
    row: pd.Series,
) -> Dict[str, str]:
    """
    Return both Pentagon diagnoses for a single row.

    Example:
        {
            "p1": "T1",
            "p2": "O"
        }
    """

    _, p1_fault, p2_fault = classify_duval_pentagon(row)

    return {
        "p1": p1_fault,
        "p2": p2_fault,
    }