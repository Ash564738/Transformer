# dga/duval_pentagon.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
import logging

logger = logging.getLogger(__name__)

SCALE = 0.75
PENTAGON_VERTICES_RAW = {
    "H2":   np.array([0.0, 40.0]),
    "C2H2": np.array([38.0, 12.4]),
    "C2H4": np.array([23.5, -32.4]),
    "CH4":  np.array([-23.5, -32.4]),
    "C2H6": np.array([-38.0, 12.4])
}
PENTAGON_VERTICES = {k: v * SCALE for k, v in PENTAGON_VERTICES_RAW.items()}

def scale_zone(zone):
    return [(x * SCALE, y * SCALE) for x, y in zone]

PENTAGON_1_ZONES = {k: scale_zone(v) for k, v in {
    "PD": [(0, 33), (-1, 33), (-1, 24.5), (0, 24.5)],
    "D1": [(0, 40), (38, 12.4), (32, -6.1), (4, 16), (0, 1.5)],
    "D2": [(4, 16), (32, -6.1), (24.3, -30), (0, -3), (0, 1.5)],
    "T3": [(0, -3), (24.3, -30), (23.5, -32.4), (1, -32.4), (-6, -4)],
    "T2": [(-6, -4), (1, -32.4), (-22.5, -32.4)],
    "T1": [(-6, -4), (-22.5, -32.4), (-23.5, -32.4), (-35, 3.1), (0, 1.5), (0, -3)],
    "S":  [(0, 1.5), (-35, 3.1), (-38, 12.4), (0, 40), (0, 33), (-1, 33), (-1, 24.5), (0, 24.5)]
}.items()}

PENTAGON_2_ZONES = {
    "PD": PENTAGON_1_ZONES["PD"],
    "D1": PENTAGON_1_ZONES["D1"],
    "D2": PENTAGON_1_ZONES["D2"],
    "S":  PENTAGON_1_ZONES["S"],
    "T3_H": scale_zone([(0, -3), (24.3, -30), (23.5, -32.4), (2.5, -32.4), (-3.5, -3)]),
    "C":    scale_zone([(-3.5, -3), (2.5, -32.4), (-21.5, -32.4), (-11, -8)]),
    "O":    scale_zone([(-3.5, -3), (-11, -8), (-21.5, -32.4), (-23.5, -32.4), (-35, 3.1), (0, 1.5), (0, -3)])
}

ZONE_COLORS = {
    "PD": "#cfff7c", "D1": "#ffffb3", "D2": "#cec9ec",
    "T1": "#9dd6ff", "T2": "#ffbd72", "T3": "#9de4d9",
    "T3_H": "#90ccc2", "C": "#fb8072", "O": "#ffd3ea", "S": "#ffcfab"
}

FAULT_EXPLANATIONS = {
    "PD": "Partial Discharge",
    "D1": "Low energy electrical discharge",
    "D2": "High energy electrical discharge (arc)",
    "T1": "Thermal fault < 300°C",
    "T2": "Thermal fault 300–700°C",
    "T3": "Thermal fault > 700°C",
    "T3_H": "Thermal fault > 700°C (oil only)",
    "C": "Carbonization of paper insulation",
    "O": "Overheating < 250°C",
    "S": "Stray Gassing"
}

PATHS_P1 = {name: Path(poly) for name, poly in PENTAGON_1_ZONES.items()}
PATHS_P2 = {name: Path(poly) for name, poly in PENTAGON_2_ZONES.items()}

def duval_pentagon_centroid(h2, ch4, c2h6, c2h4, c2h2):
    total = h2 + ch4 + c2h6 + c2h4 + c2h2
    if total <= 0 or any(np.isnan([h2, ch4, c2h6, c2h4, c2h2])):
        return None
    p_h2   = h2 / total
    p_c2h2 = c2h2 / total
    p_c2h4 = c2h4 / total
    p_ch4  = ch4 / total
    p_c2h6 = c2h6 / total
    xy = (p_h2 * PENTAGON_VERTICES["H2"] +
          p_c2h2 * PENTAGON_VERTICES["C2H2"] +
          p_c2h4 * PENTAGON_VERTICES["C2H4"] +
          p_ch4 * PENTAGON_VERTICES["CH4"] +
          p_c2h6 * PENTAGON_VERTICES["C2H6"])
    return float(xy[0]), float(xy[1])

def _find_pentagon_zone(xy, paths):
    if xy is None:
        return "UNCERTAIN"
    for zone, path in paths.items():
        if path.contains_point(xy):
            return zone
    return "UNCERTAIN"

def apply_duval_pentagon_dual(df):
    df = df.copy()
    xs, ys, faults_p1, faults_p2 = [], [], [], []
    for _, row in df.iterrows():
        h2   = row.get("h2", np.nan)
        ch4  = row.get("ch4", np.nan)
        c2h6 = row.get("c2h6", np.nan)
        c2h4 = row.get("c2h4", np.nan)
        c2h2 = row.get("c2h2", np.nan)
        total = h2 + ch4 + c2h6 + c2h4 + c2h2
        if total < 0.1:
            faults_p1.append("NORMAL")
            faults_p2.append("NORMAL")
            xs.append(0.0); ys.append(0.0)
            continue
        xy = duval_pentagon_centroid(h2, ch4, c2h6, c2h4, c2h2)
        if xy:
            xs.append(xy[0]); ys.append(xy[1])
            faults_p1.append(_find_pentagon_zone(xy, PATHS_P1))
            faults_p2.append(_find_pentagon_zone(xy, PATHS_P2))
        else:
            xs.append(np.nan); ys.append(np.nan)
            faults_p1.append("UNCERTAIN"); faults_p2.append("UNCERTAIN")
    df["p_x"] = xs; df["p_y"] = ys
    df["fault_p1"] = faults_p1; df["fault_p2"] = faults_p2

    logger.debug("Duval Pentagon dual fault applied.")
    if logger.isEnabledFor(logging.DEBUG):
        debug_cols = ["h2", "ch4", "c2h6", "c2h4", "c2h2", "p_x", "p_y", "fault_p1", "fault_p2"]
        logger.debug("Sample Pentagon results:\n" + df[debug_cols].head(5).to_string())
    return df

def apply_duval_pentagon(df: pd.DataFrame, pentagon: str = "P2") -> pd.DataFrame:
    df = apply_duval_pentagon_dual(df)
    col = "fault_p2" if str(pentagon).upper() == "P2" else "fault_p1"
    df["duval_pentagon_fault"] = df[col]
    return df