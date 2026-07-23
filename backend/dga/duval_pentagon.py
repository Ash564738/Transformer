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

def plot_duval_pentagon(ax, pentagon_type=1):
    zones = PENTAGON_1_ZONES if pentagon_type == 1 else PENTAGON_2_ZONES
    for zone_name, poly_points in zones.items():
        color = ZONE_COLORS.get(zone_name, "#FFFFFF")
        ax.add_patch(patches.Polygon(poly_points, closed=True, facecolor=color,
                                     edgecolor='none', alpha=0.85))
        poly_arr = np.array(poly_points)
        cx, cy = np.mean(poly_arr[:, 0]), np.mean(poly_arr[:, 1])
        hex_color = color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        text_color = 'white' if lum < 0.5 else 'black'
        ax.text(cx, cy, zone_name, ha='center', va='center', fontweight='bold',
                color=text_color, fontsize=9)

    outer = [PENTAGON_VERTICES[g] for g in ["H2", "C2H2", "C2H4", "CH4", "C2H6"]]
    ax.add_patch(patches.Polygon(outer, closed=True, facecolor='none',
                                 edgecolor='#2C3E50', linewidth=2))

    for gas, vertex in PENTAGON_VERTICES.items():
        ax.plot([0, vertex[0]], [0, vertex[1]], '--', color='gray', lw=0.7, alpha=0.5)

    tick_vals = [0.2, 0.4, 0.6, 0.8]
    for gas, vertex in PENTAGON_VERTICES.items():
        for k in tick_vals:
            px, py = k * vertex[0], k * vertex[1]
            ax.text(px, py, f'{int(k*100)}%', fontsize=6, color='#555555',
                    ha='center', va='center', bbox=dict(boxstyle='round,pad=0.1',
                    facecolor='white', edgecolor='none', alpha=0.7))

    label_offset = 1.12
    for gas, vertex in PENTAGON_VERTICES.items():
        lx, ly = label_offset * vertex[0], label_offset * vertex[1]
        ax.text(lx, ly, gas, fontsize=10, fontweight='bold', ha='center', va='center')

    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Duval Pentagon {pentagon_type}", fontsize=13, fontweight='bold', pad=10)

def plot_pentagon_dual(h2, ch4, c2h6, c2h4, c2h2, fault_p1=None, fault_p2=None):
    xy = duval_pentagon_centroid(h2, ch4, c2h6, c2h4, c2h2)
    if xy is None:
        return None

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4.5))
    plot_duval_pentagon(ax1, pentagon_type=1)
    plot_duval_pentagon(ax2, pentagon_type=2)

    for ax, fault in zip([ax1, ax2], [fault_p1, fault_p2]):
        ax.scatter(xy[0], xy[1], color='red', marker='X', s=80, zorder=10,
                   edgecolors='black', linewidths=1.5)
        if fault:
            ax.annotate(f"Fault: {fault}", xy=(xy[0], xy[1]), xytext=(xy[0]+2, xy[1]+2),
                        fontsize=7, color='darkred', fontweight='bold',
                        arrowprops=dict(arrowstyle='->', color='darkred'),
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred', alpha=0.9))

    plt.subplots_adjust(wspace=0.35, bottom=0.2)
    fig.suptitle("Duval Pentagon 1 & 2", fontsize=12, fontweight='bold')
    return fig