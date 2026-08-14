# dga/duval_pentagon.py
from __future__ import annotations

import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from matplotlib.path import Path

logger = logging.getLogger(__name__)


# ============================================================================
# Duval Pentagon 1 / 2
#
# Standard basis:
#     IEEE Std C57.104-2019
#
# Gas order around the Pentagon:
#
#     H2 -> C2H6 -> CH4 -> C2H4 -> C2H2
#
# The 40%-scale pentagon vertices are:
#
#     H2   = (0, 40)
#     C2H6 = (-38, 12.4)
#     CH4  = (-23.5, -32.4)
#     C2H4 = (23.5, -32.4)
#     C2H2 = (38, 12.4)
#
# The five gas values are first normalized to percentages.
# Each vertex is multiplied by the corresponding gas fraction.
# The centroid of the resulting irregular polygon is then calculated.
# ============================================================================


PENTAGON_GASES: Tuple[str, ...] = (
    "h2",
    "c2h6",
    "ch4",
    "c2h4",
    "c2h2",
)


PENTAGON_VERTICES_RAW: Dict[str, np.ndarray] = {
    "H2": np.array([0.0, 40.0], dtype=float),
    "C2H6": np.array([-38.0, 12.4], dtype=float),
    "CH4": np.array([-23.5, -32.4], dtype=float),
    "C2H4": np.array([23.5, -32.4], dtype=float),
    "C2H2": np.array([38.0, 12.4], dtype=float),
}

# Backward-compatible public alias.
PENTAGON_VERTICES = PENTAGON_VERTICES_RAW


# ============================================================================
# Zone coordinates
# ============================================================================

def _zone(
    points: Sequence[Tuple[float, float]],
) -> Tuple[Tuple[float, float], ...]:
    return tuple(
        (float(x), float(y))
        for x, y in points
    )


# IEEE C57.104-2019 Pentagon 1.
PENTAGON_1_ZONES: Dict[str, Tuple[Tuple[float, float], ...]] = {

    "PD": _zone([
        (0.0, 33.0),
        (-1.0, 33.0),
        (-1.0, 24.5),
        (0.0, 24.5),
    ]),

    "D1": _zone([
        (0.0, 40.0),
        (38.0, 12.0),
        (32.0, -6.1),
        (4.0, 16.0),
        (0.0, 1.5),
    ]),

    "D2": _zone([
        (4.0, 16.0),
        (32.0, -6.1),
        (24.3, -30.0),
        (0.0, -3.0),
        (0.0, 1.5),
    ]),

    "T3": _zone([
        (0.0, -3.0),
        (24.3, -30.0),
        (23.5, -32.4),
        (1.0, -32.4),
        (-6.0, -4.0),
    ]),

    "T2": _zone([
        (-6.0, -4.0),
        (1.0, -32.4),
        (-22.5, -32.4),
    ]),

    "T1": _zone([
        (-6.0, -4.0),
        (-22.5, -32.4),
        (-23.5, -32.4),
        (-35.0, 3.0),
        (0.0, 1.5),
        (0.0, -3.0),
    ]),

    "S": _zone([
        (0.0, 1.5),
        (-35.0, 3.1),
        (-38.0, 12.4),
        (0.0, 40.0),
        (0.0, 33.0),
        (-1.0, 33.0),
        (-1.0, 24.5),
        (0.0, 24.5),
    ]),
}


# IEEE C57.104-2019 Pentagon 2.
#
# PD / D1 / D2 / S are shared with Pentagon 1.
PENTAGON_2_ZONES: Dict[str, Tuple[Tuple[float, float], ...]] = {

    "PD": PENTAGON_1_ZONES["PD"],

    "D1": PENTAGON_1_ZONES["D1"],

    "D2": PENTAGON_1_ZONES["D2"],

    "S": PENTAGON_1_ZONES["S"],

    "T3_H": _zone([
        (0.0, -3.0),
        (24.3, -30.0),
        (23.5, -32.4),
        (2.5, -32.4),
        (-3.5, -3.0),
    ]),

    "C": _zone([
        (-3.5, -3.0),
        (2.5, -32.4),
        (-21.5, -32.4),
        (-11.0, -8.0),
    ]),

    "O": _zone([
        (-3.5, -3.0),
        (-11.0, -8.0),
        (-21.5, -32.4),
        (-23.5, -32.4),
        (-35.0, 3.1),
        (0.0, 1.5),
        (0.0, -3.0),
    ]),
}


# ============================================================================
# Metadata
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
    "PD": "Partial discharge",
    "D1": "Low-energy electrical discharge",
    "D2": "High-energy electrical discharge",
    "T1": "Low-temperature thermal fault",
    "T2": "Intermediate-temperature thermal fault",
    "T3": "High-temperature thermal fault",
    "T3_H": "High-temperature thermal fault, hydrogen-associated region",
    "C": "Paper/cellulose carbonization region",
    "O": "Thermal oil-fault region",
    "S": "Stray gassing",
}


# ============================================================================
# Validation
# ============================================================================

# This is only a numerical/data-quality gate.
# It is NOT an IEEE diagnostic threshold.
MIN_PENTAGON_TOTAL = 0.1


def _safe_gas(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError):
        return np.nan

    if not np.isfinite(value):
        return np.nan

    if value < 0:
        return np.nan

    return value


def _read_pentagon_gases(
    row: pd.Series,
) -> Optional[Tuple[float, ...]]:

    values = tuple(
        _safe_gas(
            row.get(
                gas,
                np.nan,
            )
        )
        for gas in PENTAGON_GASES
    )

    if not all(
        np.isfinite(value)
        for value in values
    ):
        return None

    return values


# ============================================================================
# Polygon centroid
# ============================================================================

def _polygon_centroid(
    points: Iterable[Tuple[float, float]],
) -> Optional[Tuple[float, float]]:

    pts = np.asarray(
        list(points),
        dtype=float,
    )

    if (
        pts.ndim != 2
        or pts.shape[0] < 3
        or pts.shape[1] != 2
    ):
        return None

    if not np.all(np.isfinite(pts)):
        return None

    x = pts[:, 0]
    y = pts[:, 1]

    x_next = np.roll(x, -1)
    y_next = np.roll(y, -1)

    cross = (
        x * y_next
        - x_next * y
    )

    signed_area = 0.5 * np.sum(cross)

    if (
        not np.isfinite(signed_area)
        or abs(signed_area) <= 1e-12
    ):
        return None

    cx = (
        np.sum(
            (x + x_next) * cross
        )
        / (6.0 * signed_area)
    )

    cy = (
        np.sum(
            (y + y_next) * cross
        )
        / (6.0 * signed_area)
    )

    if not (
        np.isfinite(cx)
        and np.isfinite(cy)
    ):
        return None

    return float(cx), float(cy)


# ============================================================================
# Centroid calculation
# ============================================================================

def duval_pentagon_centroid(
    h2: float,
    ch4: float,
    c2h6: float,
    c2h4: float,
    c2h2: float,
) -> Optional[Tuple[float, float]]:

    values = {
        "H2": _safe_gas(h2),
        "C2H6": _safe_gas(c2h6),
        "CH4": _safe_gas(ch4),
        "C2H4": _safe_gas(c2h4),
        "C2H2": _safe_gas(c2h2),
    }

    if not all(
        np.isfinite(value)
        for value in values.values()
    ):
        return None

    total = sum(values.values())

    if (
        not np.isfinite(total)
        or total <= 0
        or total < MIN_PENTAGON_TOTAL
    ):
        return None

    fractions = {
        gas: value / total
        for gas, value in values.items()
    }

    # IMPORTANT:
    # Build the five scaled points in standard gas order.
    points = []

    for gas in (
        "H2",
        "C2H6",
        "CH4",
        "C2H4",
        "C2H2",
    ):
        vertex = PENTAGON_VERTICES[gas]
        fraction = fractions[gas]

        points.append(
            (
                float(vertex[0] * fraction),
                float(vertex[1] * fraction),
            )
        )

    return _polygon_centroid(points)


# ============================================================================
# Boundary-aware zone classification
# ============================================================================

def _point_on_segment(
    point: Tuple[float, float],
    a: Tuple[float, float],
    b: Tuple[float, float],
    tolerance: float = 1e-8,
) -> bool:

    px, py = point
    ax, ay = a
    bx, by = b

    abx = bx - ax
    aby = by - ay

    apx = px - ax
    apy = py - ay

    cross = (
        abx * apy
        - aby * apx
    )

    if abs(cross) > tolerance:
        return False

    dot = (
        apx * abx
        + apy * aby
    )

    if dot < -tolerance:
        return False

    ab_squared = (
        abx * abx
        + aby * aby
    )

    if dot > ab_squared + tolerance:
        return False

    return True


def _point_on_polygon_boundary(
    point: Tuple[float, float],
    polygon: Sequence[Tuple[float, float]],
    tolerance: float = 1e-8,
) -> bool:

    for i in range(len(polygon)):
        a = polygon[i]
        b = polygon[(i + 1) % len(polygon)]

        if _point_on_segment(
            point,
            a,
            b,
            tolerance=tolerance,
        ):
            return True

    return False


def _find_pentagon_zone(
    xy: Optional[Tuple[float, float]],
    zones: Dict[str, Tuple[Tuple[float, float], ...]],
    paths: Dict[str, Path],
) -> str:

    if xy is None:
        return "ABSTAIN"

    x, y = xy

    if not (
        np.isfinite(x)
        and np.isfinite(y)
    ):
        return "ABSTAIN"

    point = (
        float(x),
        float(y),
    )

    # --------------------------------------------------------
    # Do not silently assign a boundary point to whichever
    # polygon happens to appear first in the dictionary.
    # --------------------------------------------------------

    boundary_hits = [
        zone
        for zone, polygon in zones.items()
        if _point_on_polygon_boundary(
            point,
            polygon,
        )
    ]

    if len(boundary_hits) > 1:
        return "ABSTAIN"

    if len(boundary_hits) == 1:
        return boundary_hits[0]

    # --------------------------------------------------------
    # Strictly inside polygon.
    # --------------------------------------------------------

    hits = [
        zone
        for zone, path in paths.items()
        if path.contains_point(
            point,
            radius=0.0,
        )
    ]

    if len(hits) == 1:
        return hits[0]

    # Overlap should never be resolved by dictionary order.
    if len(hits) > 1:
        logger.warning(
            "Duval Pentagon point %.10f, %.10f falls in multiple zones: %s",
            x,
            y,
            hits,
        )

    return "ABSTAIN"


# ============================================================================
# Paths
# ============================================================================

PATHS_P1: Dict[str, Path] = {
    name: Path(
        np.asarray(
            polygon,
            dtype=float,
        )
    )
    for name, polygon in PENTAGON_1_ZONES.items()
}


PATHS_P2: Dict[str, Path] = {
    name: Path(
        np.asarray(
            polygon,
            dtype=float,
        )
    )
    for name, polygon in PENTAGON_2_ZONES.items()
}


# ============================================================================
# Single-row classification
# ============================================================================

def classify_duval_pentagon(
    row: pd.Series,
) -> Tuple[
    Optional[Tuple[float, float]],
    str,
    str,
]:

    gases = _read_pentagon_gases(row)

    if gases is None:
        return (
            None,
            "ABSTAIN",
            "ABSTAIN",
        )

    h2, c2h6, ch4, c2h4, c2h2 = gases

    total = (
        h2
        + c2h6
        + ch4
        + c2h4
        + c2h2
    )

    if (
        not np.isfinite(total)
        or total < MIN_PENTAGON_TOTAL
    ):
        return (
            None,
            "ABSTAIN",
            "ABSTAIN",
        )

    xy = duval_pentagon_centroid(
        h2=h2,
        ch4=ch4,
        c2h6=c2h6,
        c2h4=c2h4,
        c2h2=c2h2,
    )

    if xy is None:
        return (
            None,
            "ABSTAIN",
            "ABSTAIN",
        )

    fault_p1 = _find_pentagon_zone(
        xy,
        PENTAGON_1_ZONES,
        PATHS_P1,
    )

    fault_p2 = _find_pentagon_zone(
        xy,
        PENTAGON_2_ZONES,
        PATHS_P2,
    )

    return (
        xy,
        fault_p1,
        fault_p2,
    )


# ============================================================================
# Apply both pentagons
# ============================================================================

def apply_duval_pentagon_dual(
    df: pd.DataFrame,
) -> pd.DataFrame:

    df = df.copy()

    xs = []
    ys = []

    p1_faults = []
    p2_faults = []

    for _, row in df.iterrows():

        xy, p1_fault, p2_fault = (
            classify_duval_pentagon(row)
        )

        if xy is None:
            xs.append(np.nan)
            ys.append(np.nan)
        else:
            xs.append(float(xy[0]))
            ys.append(float(xy[1]))

        p1_faults.append(p1_fault)
        p2_faults.append(p2_fault)

    df["p_x"] = xs
    df["p_y"] = ys

    df["duval_pentagon_p1_fault"] = p1_faults
    df["duval_pentagon_p2_fault"] = p2_faults

    # Backward-compatible columns.
    df["fault_p1"] = p1_faults
    df["fault_p2"] = p2_faults

    logger.debug(
        "Duval Pentagon 1 and 2 applied."
    )

    return df


# ============================================================================
# Backward-compatible API
# ============================================================================

def apply_duval_pentagon(
    df: pd.DataFrame,
    pentagon: str = "P2",
) -> pd.DataFrame:
    """
    Calculate both Pentagon 1 and Pentagon 2.

    The requested pentagon only determines the legacy
    `duval_pentagon_fault` output.

    P1 and P2 remain separate outputs because they are two
    interpretations of the same Duval Pentagon centroid.
    """

    df = apply_duval_pentagon_dual(df)

    pentagon = str(
        pentagon
    ).strip().upper()

    if pentagon == "P1":
        df["duval_pentagon_fault"] = (
            df["duval_pentagon_p1_fault"]
        )
    else:
        df["duval_pentagon_fault"] = (
            df["duval_pentagon_p2_fault"]
        )

    return df


# ============================================================================
# Convenience API
# ============================================================================

def get_duval_pentagon_faults(
    row: pd.Series,
) -> Dict[str, str]:

    _, p1, p2 = classify_duval_pentagon(row)

    return {
        "p1": p1,
        "p2": p2,
    }