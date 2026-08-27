# dga_features.py
from __future__ import annotations

"""Canonical scale-invariant DGA representations.

The feature set is deliberately limited to quantities directly derived from
the five hydrocarbon gases used by Duval Triangle/Pentagon and ratio methods.
It introduces no learned or hand-assigned severity weight.

Feature families:
- five-gas percentage composition (pentagon basis);
- three-gas percentage composition (triangle basis);
- three diagnostic ratios used by IEC/Rogers-style interpretation;
- Duval Pentagon Cartesian centroid coordinates.

Undefined ratios remain NaN and are handled by the downstream model imputer.
This avoids silently converting an undefined physical ratio into a real zero.
"""

import numpy as np
import pandas as pd
from matplotlib.path import Path

GAS5 = ["h2", "c2h6", "ch4", "c2h4", "c2h2"]
TRI_GASES = ["ch4", "c2h4", "c2h2"]

PENTAGON_ANGLES_DEG = {
    "h2": 90.0,
    "c2h6": 162.0,
    "ch4": 234.0,
    "c2h4": 306.0,
    "c2h2": 18.0,
}

FEATS = [
    *(f"pct5_{g}" for g in GAS5),
    "pct_ch4_tri", "pct_c2h4_tri", "pct_c2h2_tri",
    "ratio_ch4_h2", "ratio_c2h2_c2h4", "ratio_c2h4_c2h6",
    "pent_cx", "pent_cy",
]

# Same outer-pentagon geometry used by the teammate implementation.  The
# centroid is a deterministic transform of the measured five-gas composition.
PENTAGON_ZONE_POLYGONS = {
    "PD": [(0.0, 33.0), (-1.0, 33.0), (-1.0, 24.5), (0.0, 24.5)],
    "D1": [(0.0, 40.0), (38.0, 12.4), (32.0, -6.0), (4.0, 16.0), (0.0, 1.5)],
    "D2": [(4.0, 16.0), (32.0, -6.0), (24.0, -30.0), (-1.0, -2.0)],
    "T3": [(-1.0, -2.0), (-6.0, -4.0), (1.0, -32.4), (-23.3, -32.4), (24.0, -30.0)],
    "T2": [(1.0, -32.4), (-22.5, -32.4), (-6.0, -4.0)],
    "T1": [(0.0, 1.5), (-35.0, 3.0), (-23.3, -32.4), (-22.5, -32.4), (-6.0, -4.0), (-1.0, -2.0)],
    "S": [(0.0, 1.5), (0.0, 24.5), (-1.0, 24.5), (-1.0, 33.0), (0.0, 33.0),
           (0.0, 40.0), (-38.0, 12.4), (-35.0, 3.0)],
}
ZONE_PATHS = {name: Path(np.asarray(points, dtype=float)) for name, points in PENTAGON_ZONE_POLYGONS.items()}


def polygon_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    pts = np.asarray(points + [points[0]], dtype=float)
    x, y = pts[:, 0], pts[:, 1]
    cross = x[:-1] * y[1:] - x[1:] * y[:-1]
    area = cross.sum() / 2.0
    if abs(area) < 1e-12:
        return float(np.mean(x[:-1])), float(np.mean(y[:-1]))
    cx = ((x[:-1] + x[1:]) * cross).sum() / (6.0 * area)
    cy = ((y[:-1] + y[1:]) * cross).sum() / (6.0 * area)
    return float(cx), float(cy)


def _numeric(df: pd.DataFrame, col: str) -> pd.Series:
    if col not in df.columns:
        return pd.Series(np.nan, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def build_scale_invariant_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame(index=df.index)
    gas = {g: _numeric(df, g).clip(lower=0) for g in GAS5}

    total5 = sum(gas.values()).replace(0.0, np.nan)
    for g in GAS5:
        out[f"pct5_{g}"] = gas[g] / total5 * 100.0

    tri_total = (gas["ch4"] + gas["c2h4"] + gas["c2h2"]).replace(0.0, np.nan)
    out["pct_ch4_tri"] = gas["ch4"] / tri_total * 100.0
    out["pct_c2h4_tri"] = gas["c2h4"] / tri_total * 100.0
    out["pct_c2h2_tri"] = gas["c2h2"] / tri_total * 100.0

    # Undefined ratio = NaN, not an artificial epsilon-shifted value.
    out["ratio_ch4_h2"] = gas["ch4"] / gas["h2"].replace(0.0, np.nan)
    out["ratio_c2h2_c2h4"] = gas["c2h2"] / gas["c2h4"].replace(0.0, np.nan)
    out["ratio_c2h4_c2h6"] = gas["c2h4"] / gas["c2h6"].replace(0.0, np.nan)

    cx_values: list[float] = []
    cy_values: list[float] = []
    for idx in df.index:
        if any(pd.isna(gas[g].loc[idx]) for g in GAS5):
            cx_values.append(np.nan)
            cy_values.append(np.nan)
            continue
        vertices = []
        for g in GAS5:
            r = float(gas[g].loc[idx] / total5.loc[idx] * 100.0) if pd.notna(total5.loc[idx]) else np.nan
            angle = np.deg2rad(PENTAGON_ANGLES_DEG[g])
            vertices.append((r * np.cos(angle), r * np.sin(angle)))
        if not np.isfinite(np.asarray(vertices, dtype=float)).all():
            cx_values.append(np.nan)
            cy_values.append(np.nan)
            continue
        cx, cy = polygon_centroid(vertices)
        cx_values.append(cx)
        cy_values.append(cy)

    out["pent_cx"] = cx_values
    out["pent_cy"] = cy_values
    return out[FEATS].replace([np.inf, -np.inf], np.nan)


def classify_pentagon_from_centroid(cx: float, cy: float) -> str:
    if not (np.isfinite(cx) and np.isfinite(cy)):
        return "ND"
    for name, path in ZONE_PATHS.items():
        if path.contains_point((cx, cy), radius=1e-6):
            return name
    # Explicitly return ND for points outside the valid polygon instead of
    # forcing a nearest-zone class that would manufacture a label.
    return "ND"


__all__ = ["GAS5", "FEATS", "build_scale_invariant_features", "classify_pentagon_from_centroid"]