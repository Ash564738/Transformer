# duval_pentagon.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path

# ==========================================================
# 1. TỌA ĐỘ GỐC 5 ĐỈNH NGŨ GIÁC DUVAL (R=40)
# ==========================================================
SCALE = 0.75
PENTAGON_VERTICES_RAW = {
    "H2":   np.array([0.0, 40.0]),
    "C2H2": np.array([38.0, 12.4]),
    "C2H4": np.array([23.5, -32.4]),
    "CH4":  np.array([-23.5, -32.4]),
    "C2H6": np.array([-38.0, 12.4])
}
PENTAGON_VERTICES = {k: v * SCALE for k, v in PENTAGON_VERTICES_RAW.items()}

# ==========================================================
# 2. ĐỊNH NGHĨA CÁC VÙNG LỖI (PENTAGON 1 & 2) – scale đúng 1 lần
# ==========================================================
def scale_zone(zone):
    return [(x * SCALE, y * SCALE) for x, y in zone]

# Pentagon 1
PENTAGON_1_ZONES = {k: scale_zone(v) for k, v in {
    "PD": [(0, 33), (-1, 33), (-1, 24.5), (0, 24.5)],
    "D1": [(0, 40), (38, 12.4), (32, -6.1), (4, 16), (0, 1.5)],
    "D2": [(4, 16), (32, -6.1), (24.3, -30), (0, -3), (0, 1.5)],
    "T3": [(0, -3), (24.3, -30), (23.5, -32.4), (1, -32.4), (-6, -4)],
    "T2": [(-6, -4), (1, -32.4), (-22.5, -32.4)],
    "T1": [(-6, -4), (-22.5, -32.4), (-23.5, -32.4), (-35, 3.1), (0, 1.5), (0, -3)],
    "S":  [(0, 1.5), (-35, 3.1), (-38, 12.4), (0, 40), (0, 33), (-1, 33), (-1, 24.5), (0, 24.5)]
}.items()}

# Pentagon 2 (kế thừa vùng chung từ P1, thêm vùng riêng)
PENTAGON_2_ZONES = {
    "PD": PENTAGON_1_ZONES["PD"],
    "D1": PENTAGON_1_ZONES["D1"],
    "D2": PENTAGON_1_ZONES["D2"],
    "S":  PENTAGON_1_ZONES["S"],
    "T3-H": scale_zone([(0, -3), (24.3, -30), (23.5, -32.4), (2.5, -32.4), (-3.5, -3)]),
    "C":    scale_zone([(-3.5, -3), (2.5, -32.4), (-21.5, -32.4), (-11, -8)]),
    "O":    scale_zone([(-3.5, -3), (-11, -8), (-21.5, -32.4), (-23.5, -32.4), (-35, 3.1), (0, 1.5), (0, -3)])
}

# Màu sắc tươi
ZONE_COLORS = {
    "PD": "#4daf4a", "D1": "#ff7f00", "D2": "#377eb8",
    "T1": "#f781bf", "T2": "#a65628", "T3": "#984ea3",
    "T3-H": "#984ea3", "C": "#e41a1c", "O": "#ffd92f", "S": "#b3b3b3"
}

FAULT_EXPLANATIONS = {
    "PD": "Partial Discharge",
    "D1": "Low energy electrical discharge",
    "D2": "High energy electrical discharge (arc)",
    "T1": "Thermal fault < 300°C",
    "T2": "Thermal fault 300–700°C",
    "T3": "Thermal fault > 700°C",
    "T3-H": "Thermal fault > 700°C (oil only)",
    "C": "Carbonization of paper insulation",
    "O": "Overheating < 250°C",
    "S": "Stray Gassing"
}

PATHS_P1 = {name: Path(poly) for name, poly in PENTAGON_1_ZONES.items()}
PATHS_P2 = {name: Path(poly) for name, poly in PENTAGON_2_ZONES.items()}

# ==========================================================
# 3. THUẬT TOÁN ĐỊNH VỊ VÀ TÌM KIẾM VÙNG LỖI
# ==========================================================
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
    if paths is None or xy is None:
        return "INVALID_LOW_GAS"
    for zone, path in paths.items():
        if path.contains_point(xy):
            return zone
    x, y = xy
    best_zone, best_dist = None, np.inf
    for zone, path in paths.items():
        vertices = path.vertices
        d = np.min(np.sqrt((vertices[:, 0] - x) ** 2 + (vertices[:, 1] - y) ** 2))
        if d < best_dist:
            best_dist = d
            best_zone = zone
    return best_zone if best_zone is not None else "UNCERTAIN"

def apply_duval_pentagon_dual(df):
    df = df.copy()
    xs, ys, faults_p1, faults_p2 = [], [], [], []
    for _, row in df.iterrows():
        h2   = row.get("h2", np.nan)
        ch4  = row.get("ch4", np.nan)
        c2h6 = row.get("c2h6", np.nan)
        c2h4 = row.get("c2h4", np.nan)
        c2h2 = row.get("c2h2", np.nan)
        xy = duval_pentagon_centroid(h2, ch4, c2h6, c2h4, c2h2)
        if xy:
            xs.append(xy[0]); ys.append(xy[1])
            faults_p1.append(_find_pentagon_zone(xy, PATHS_P1))
            faults_p2.append(_find_pentagon_zone(xy, PATHS_P2))
        else:
            xs.append(np.nan); ys.append(np.nan)
            faults_p1.append("INVALID_LOW_GAS"); faults_p2.append("INVALID_LOW_GAS")
    df["p_x"] = xs; df["p_y"] = ys
    df["fault_p1"] = faults_p1; df["fault_p2"] = faults_p2
    return df

def apply_duval_pentagon(df: pd.DataFrame, pentagon: str = "P2") -> pd.DataFrame:
    df = apply_duval_pentagon_dual(df)
    col = "fault_p2" if str(pentagon).upper() == "P2" else "fault_p1"
    df["duval_pentagon_fault"] = df[col]
    return df

# ==========================================================
# 4. HÀM VẼ PENTAGON
# ==========================================================
def plot_duval_pentagon(ax, pentagon_type=1):
    zones = PENTAGON_1_ZONES if pentagon_type == 1 else PENTAGON_2_ZONES

    for zone_name, poly_points in zones.items():
        color = ZONE_COLORS.get(zone_name, "#FFFFFF")
        ax.add_patch(patches.Polygon(poly_points, closed=True, facecolor=color,
                                     edgecolor='none', alpha=0.85))
        poly_arr = np.array(poly_points)
        cx, cy = np.mean(poly_arr[:, 0]), np.mean(poly_arr[:, 1])
        # Chọn màu chữ tương phản
        hex_color = color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        lum = (0.299 * r + 0.587 * g + 0.114 * b) / 255
        text_color = 'white' if lum < 0.5 else 'black'
        ax.text(cx, cy, zone_name, ha='center', va='center', fontweight='bold',
                color=text_color, fontsize=9)

    # Viền ngũ giác ngoài
    outer = [PENTAGON_VERTICES[g] for g in ["H2", "C2H2", "C2H4", "CH4", "C2H6"]]
    ax.add_patch(patches.Polygon(outer, closed=True, facecolor='none',
                                 edgecolor='#2C3E50', linewidth=2))

    # Trục từ tâm ra đỉnh (nét đứt)
    for gas, vertex in PENTAGON_VERTICES.items():
        ax.plot([0, vertex[0]], [0, vertex[1]], '--', color='gray', alpha=0.4)

    # Nhãn khí ở ngoài đỉnh
    label_offset = 1.12
    for gas, vertex in PENTAGON_VERTICES.items():
        lx, ly = label_offset * vertex[0], label_offset * vertex[1]
        ax.text(lx, ly, gas, fontsize=10, fontweight='bold', ha='center', va='center')

    ax.set_xlim(-35, 35)
    ax.set_ylim(-35, 35)
    ax.set_aspect('equal')
    ax.axis('off')
    ax.set_title(f"Duval Pentagon {pentagon_type}", fontsize=13, fontweight='bold', pad=10)

def plot_pentagon_with_point(ax, pentagon_type, centroid_xy, fault_name, gas_info=""):
    plot_duval_pentagon(ax, pentagon_type)
    if centroid_xy is not None:
        ax.scatter(*centroid_xy, color='red', marker='X', s=140, zorder=10,
                   edgecolors='black', linewidths=1.5)
        ax.annotate(f'Fault: {fault_name}\n{gas_info}',
                    xy=centroid_xy, xytext=(centroid_xy[0]+3, centroid_xy[1]+3),
                    fontsize=8, color='darkred', fontweight='bold',
                    arrowprops=dict(arrowstyle='->', color='darkred'),
                    bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred', alpha=0.9))
    zones = PENTAGON_1_ZONES if pentagon_type == 1 else PENTAGON_2_ZONES
    legend_handles = [plt.Rectangle((0,0),1,1, facecolor=ZONE_COLORS[z], alpha=0.85,
                                    label=f"{z}: {FAULT_EXPLANATIONS.get(z, z)}")
                      for z in zones]
    ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
              fontsize=7, framealpha=0.9, title="Fault Zones")

# ==========================================================
# 5. DEMO
# ==========================================================
if __name__ == "__main__":
    sample_data = {
        "transformer_id": ["TX001"],
        "h2": [50.0],
        "ch4": [160.0],
        "c2h6": [20.0],
        "c2h4": [30.0],
        "c2h2": [90.0]
    }
    df = pd.DataFrame(sample_data)
    df_analyzed = apply_duval_pentagon_dual(df)

    x = df_analyzed["p_x"].iloc[0]
    y = df_analyzed["p_y"].iloc[0]
    f1 = df_analyzed["fault_p1"].iloc[0]
    f2 = df_analyzed["fault_p2"].iloc[0]
    gas_info = f"H2:{df['h2'].iloc[0]} CH4:{df['ch4'].iloc[0]} C2H6:{df['c2h6'].iloc[0]} C2H4:{df['c2h4'].iloc[0]} C2H2:{df['c2h2'].iloc[0]}"

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
    plot_pentagon_with_point(ax1, 1, (x, y), f1, gas_info)
    plot_pentagon_with_point(ax2, 2, (x, y), f2, gas_info)
    plt.subplots_adjust(wspace=0.3)
    plt.show()