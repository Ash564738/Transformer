# dga/duval_triangle.py

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
from matplotlib.path import Path

logger = logging.getLogger(__name__)

SQRT3 = np.sqrt(3.0)
H = SQRT3 / 2.0

# ============================================================
# 1. TERNARY COORDINATE CONVERSION
# ============================================================

def ternary_to_xy(
    ch4: float,
    c2h2: float,
    c2h4: float,
) -> tuple[float, float] | None:
    """
    Convert Duval Triangle gas percentages to Cartesian XY.

    Input values may be raw concentrations or percentages.
    The function normalizes them internally.

    Returns:
        (x, y) if valid
        None if total <= 0 or any value is invalid
    """

    values = np.asarray(
        [ch4, c2h2, c2h4],
        dtype=float,
    )

    if not np.all(np.isfinite(values)):
        return None

    if np.any(values < 0):
        return None

    total = float(values.sum())

    if total <= 0:
        return None

    ch4_n, c2h2_n, c2h4_n = values / total

    # Standard ternary transformation:
    #
    # CH4  -> top vertex
    # C2H4 -> bottom-right
    # C2H2 -> bottom-left
    #
    # x = C2H4 + 0.5 * CH4
    # y = sqrt(3)/2 * CH4

    x = float(c2h4_n + 0.5 * ch4_n)
    y = float(H * ch4_n)

    return x, y


def build_polygon_from_verts(coords) -> Path:
    """
    Build a matplotlib Path from polygon coordinates.
    """

    verts = np.asarray(coords, dtype=float)

    if len(verts) < 3:
        raise ValueError("A polygon requires at least 3 vertices.")

    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack([verts, verts[0]])

    codes = (
        [Path.MOVETO]
        + [Path.LINETO] * (len(verts) - 2)
        + [Path.CLOSEPOLY]
    )

    return Path(verts, codes)


# ============================================================
# 2. DUVAL TRIANGLE ZONES
# ============================================================

REGION_COORDS = {
    "PD": {
        "a": [98, 100, 98],
        "b": [0, 0, 2],
        "c": [2, 0, 0],
    },

    "D1": {
        "a": [0, 0, 64, 87],
        "b": [100, 77, 13, 13],
        "c": [0, 23, 23, 0],
    },

    "D2": {
        "a": [0, 0, 31, 47, 64],
        "b": [77, 29, 29, 13, 13],
        "c": [23, 71, 40, 40, 23],
    },

    "DT": {
        "a": [0, 0, 35, 46, 96, 87, 47, 31],
        "b": [29, 15, 15, 4, 4, 13, 13, 29],
        "c": [71, 85, 50, 50, 0, 0, 40, 40],
    },

    "T1": {
        "a": [76, 80, 98, 98, 96],
        "b": [4, 0, 0, 2, 4],
        "c": [20, 20, 2, 0, 0],
    },

    "T2": {
        "a": [46, 50, 80, 76],
        "b": [4, 0, 0, 4],
        "c": [50, 50, 20, 20],
    },

    "T3": {
        "a": [0, 0, 50, 35],
        "b": [15, 0, 0, 15],
        "c": [85, 100, 50, 50],
    },
}


PATHS_T1: dict[str, Path] = {}

for zone, coords in REGION_COORDS.items():

    verts_xy = []

    for ch4, c2h2, c2h4 in zip(
        coords["a"],
        coords["b"],
        coords["c"],
    ):
        xy = ternary_to_xy(
            ch4,
            c2h2,
            c2h4,
        )

        if xy is not None:
            verts_xy.append(xy)

    if len(verts_xy) >= 3:
        PATHS_T1[zone] = build_polygon_from_verts(
            verts_xy
        )


ZONE_COLORS = {
    "PD": "#b3de69",
    "T1": "#80b1d3",
    "T2": "#fdb462",
    "T3": "#8dd3c7",
    "D1": "#ffffb3",
    "D2": "#bebada",
    "DT": "#fb8072",
}


ZONE_SHORT_LABELS = {
    "PD": "PD",
    "T1": "T1",
    "T2": "T2",
    "T3": "T3",
    "D1": "D1",
    "D2": "D2",
    "DT": "DT",
}


# ============================================================
# 3. SAFE GAS READING
# ============================================================

def _safe_gas(value) -> float:
    """
    Convert a gas value to a safe non-negative finite float.

    Missing, non-numeric, NaN, inf and negative values are
    treated as invalid and return NaN.
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


# ============================================================
# 4. DUVAL TRIANGLE DIAGNOSIS
# ============================================================

def duval_triangle_1(
    ch4: float,
    c2h4: float,
    c2h2: float,
) -> str:
    """
    Duval Triangle 1 diagnostic classification.

    Returns one of:
        PD
        D1
        D2
        DT
        T1
        T2
        T3
        ABSTAIN

    Important:
        This method does NOT return NORMAL.

    Duval Triangle is a fault classification method.
    Insufficient or non-classifiable data results in ABSTAIN.
    """

    ch4 = _safe_gas(ch4)
    c2h4 = _safe_gas(c2h4)
    c2h2 = _safe_gas(c2h2)

    values = np.asarray(
        [ch4, c2h4, c2h2],
        dtype=float,
    )

    # --------------------------------------------------------
    # Invalid / missing input
    # --------------------------------------------------------

    if not np.all(np.isfinite(values)):
        return "ABSTAIN"

    # --------------------------------------------------------
    # Very low total concentration
    #
    # Do NOT return NORMAL.
    # There is insufficient basis for the Triangle method
    # to establish a fault or a healthy condition.
    # --------------------------------------------------------

    total = float(values.sum())

    if total < 0.1:
        return "ABSTAIN"

    # --------------------------------------------------------
    # Calculate ternary coordinate.
    #
    # Use raw concentrations directly.
    # ternary_to_xy() performs normalization internally.
    # --------------------------------------------------------

    xy = ternary_to_xy(
        ch4,
        c2h2,
        c2h4,
    )

    if xy is None:
        return "ABSTAIN"

    # --------------------------------------------------------
    # Determine zone.
    #
    # Do not force a result when the point lies outside the
    # defined diagnostic regions.
    # --------------------------------------------------------

    for zone, path in PATHS_T1.items():

        if path.contains_point(
            xy,
            radius=1e-9,
        ):
            return zone

    return "ABSTAIN"


# ============================================================
# 5. APPLY TO DATAFRAME
# ============================================================

def apply_duval_triangle(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Apply Duval Triangle 1 classification to a DataFrame.

    Output columns:

        t_x
        t_y
        duval_triangle_fault

    The method never generates NORMAL.

    Invalid / insufficient / out-of-zone samples are marked
    as ABSTAIN.
    """

    df = df.copy()

    xs: list[float] = []
    ys: list[float] = []
    faults: list[str] = []

    for _, row in df.iterrows():

        ch4 = _safe_gas(
            row.get("ch4", np.nan)
        )

        c2h4 = _safe_gas(
            row.get("c2h4", np.nan)
        )

        c2h2 = _safe_gas(
            row.get("c2h2", np.nan)
        )

        values = np.asarray(
            [ch4, c2h4, c2h2],
            dtype=float,
        )

        # ----------------------------------------------------
        # Invalid sample
        # ----------------------------------------------------

        if not np.all(np.isfinite(values)):

            xs.append(np.nan)
            ys.append(np.nan)
            faults.append("ABSTAIN")

            continue

        total = float(values.sum())

        # ----------------------------------------------------
        # Insufficient gas concentration
        #
        # IMPORTANT:
        # Never convert this into NORMAL.
        # ----------------------------------------------------

        if total < 0.1:

            xs.append(np.nan)
            ys.append(np.nan)
            faults.append("ABSTAIN")

            continue

        # ----------------------------------------------------
        # Calculate point
        # ----------------------------------------------------

        xy = ternary_to_xy(
            ch4,
            c2h2,
            c2h4,
        )

        if xy is None:

            xs.append(np.nan)
            ys.append(np.nan)
            faults.append("ABSTAIN")

            continue

        xs.append(xy[0])
        ys.append(xy[1])

        # ----------------------------------------------------
        # Classify
        # ----------------------------------------------------

        fault = duval_triangle_1(
            ch4,
            c2h4,
            c2h2,
        )

        faults.append(fault)

    # --------------------------------------------------------
    # Output
    # --------------------------------------------------------

    df["t_x"] = xs
    df["t_y"] = ys
    df["duval_triangle_fault"] = faults

    logger.debug(
        "Duval Triangle 1 fault classification applied."
    )

    if logger.isEnabledFor(logging.DEBUG):

        cols = [
            "ch4",
            "c2h4",
            "c2h2",
            "t_x",
            "t_y",
            "duval_triangle_fault",
        ]

        available_cols = [
            col
            for col in cols
            if col in df.columns
        ]

        logger.debug(
            "Sample Duval Triangle results:\n%s",
            df[available_cols]
            .head(5)
            .to_string(),
        )

    return df