# dga/duval_triangle.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch
import logging

logger = logging.getLogger(__name__)

SQRT3 = np.sqrt(3)
h = SQRT3 / 2.0

# ========== 1. HÀM CHUYỂN ĐỔI TỌA ĐỘ ==========
def ternary_to_xy(ch4, c2h2, c2h4):
    total = ch4 + c2h2 + c2h4
    if total <= 0:
        return None
    ch4, c2h2, c2h4 = [v / total for v in (ch4, c2h2, c2h4)]
    x = c2h4 + 0.5 * ch4
    y = h * ch4
    return (x, y)

def build_polygon_from_verts(coords):
    verts = np.asarray(coords)
    if not np.allclose(verts[0], verts[-1]):
        verts = np.vstack([verts, verts[0]])
    codes = [Path.MOVETO] + [Path.LINETO] * (len(verts) - 2) + [Path.CLOSEPOLY]
    return Path(verts, codes)

# ========== 2. VÙNG LỖI ==========
REGION_COORDS = {
    "PD": {"a": [98, 100, 98], "b": [0,   0,  2], "c": [2,   0,  0]},
    "D1": {"a": [0,   0, 64, 87], "b": [100,77, 13, 13], "c": [0,  23, 23,  0]},
    "D2": {"a": [0,   0, 31, 47, 64], "b": [77, 29, 29, 13, 13], "c": [23, 71, 40, 40, 23]},
    "DT": {"a": [0,   0, 35, 46, 96, 87, 47, 31],
           "b": [29, 15, 15,  4,  4, 13, 13, 29],
           "c": [71, 85, 50, 50,  0,  0, 40, 40]},
    "T1": {"a": [76, 80, 98, 98, 96], "b": [4,  0,  0,  2,  4], "c": [20, 20,  2,  0,  0]},
    "T2": {"a": [46, 50, 80, 76], "b": [4,  0,  0,  4], "c": [50, 50, 20, 20]},
    "T3": {"a": [0,   0, 50, 35], "b": [15,  0,  0, 15], "c": [85,100, 50, 50]},
}

PATHS_T1 = {}
for zone, coords in REGION_COORDS.items():
    verts_xy = []
    for ch4, c2h2, c2h4 in zip(coords["a"], coords["b"], coords["c"]):
        xy = ternary_to_xy(ch4, c2h2, c2h4)
        if xy is not None:
            verts_xy.append(xy)
    if len(verts_xy) >= 3:
        PATHS_T1[zone] = build_polygon_from_verts(verts_xy)

zone_colors = {
    "PD": "#b3de69", "T1": "#80b1d3", "T2": "#fdb462",
    "T3": "#8dd3c7", "D1": "#ffffb3", "D2": "#bebada", "DT": "#fb8072",
}
zone_short_labels = {"PD": "PD", "T1": "T1", "T2": "T2", "T3": "T3", "D1": "D1", "D2": "D2", "DT": "DT"}

# ========== 3. CHẨN ĐOÁN ==========
def duval_triangle_1(ch4, c2h4, c2h2):
    if any(np.isnan(x) or x < 0 for x in [ch4, c2h4, c2h2]):
        return "UNCERTAIN"
    total = ch4 + c2h4 + c2h2
    if total < 0.1:
        return "NORMAL"
    if total == 0:
        return "UNCERTAIN"
    pCH4  = ch4  / total * 100
    pC2H4 = c2h4 / total * 100
    pC2H2 = c2h2 / total * 100
    xy = ternary_to_xy(pCH4, pC2H2, pC2H4)
    if xy is None:
        return "UNCERTAIN"
    x, y = xy
    for zone, path in PATHS_T1.items():
        if path.contains_point((x, y), radius=1e-6):
            return zone
    return "UNCERTAIN"

# ========== 4. ÁP DỤNG CHO DATAFRAME ==========
def apply_duval_triangle(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    xs, ys, faults = [], [], []
    for _, row in df.iterrows():
        ch4 = row.get("ch4", np.nan)
        c2h4 = row.get("c2h4", np.nan)
        c2h2 = row.get("c2h2", np.nan)
        total = ch4 + c2h4 + c2h2
        if total > 0:
            pCH4 = ch4 / total * 100
            pC2H4 = c2h4 / total * 100
            pC2H2 = c2h2 / total * 100
            xy = ternary_to_xy(pCH4, pC2H2, pC2H4)
            if xy:
                xs.append(xy[0]); ys.append(xy[1])
                faults.append(duval_triangle_1(ch4, c2h4, c2h2))
            else:
                xs.append(np.nan); ys.append(np.nan); faults.append("UNCERTAIN")
        else:
            xs.append(np.nan); ys.append(np.nan)
            faults.append("NORMAL" if total < 0.1 else "UNCERTAIN")
    df["t_x"] = xs
    df["t_y"] = ys
    df["duval_triangle_fault"] = faults

    logger.debug("Duval Triangle fault applied.")
    if logger.isEnabledFor(logging.DEBUG):
        cols = ["ch4", "c2h4", "c2h2", "t_x", "t_y", "duval_triangle_fault"]
        logger.debug("Sample Duval Triangle results:\n" + df[cols].head(5).to_string())
    return df