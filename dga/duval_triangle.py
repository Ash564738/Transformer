# duval_triangle.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import PathPatch, FancyArrowPatch

SQRT3 = np.sqrt(3)
h = SQRT3 / 2.0

# ========== 1. DỮ LIỆU MẪU ==========
H2   = 50
CH4  = 160
C2H6 = 20
C2H4 = 30
C2H2 = 90

# ========== 2. HÀM CHUYỂN ĐỔI (đỉnh CH4, góc phải C2H4, góc trái C2H2) ==========
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

# ========== 3. VÙNG LỖI ==========
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

# ========== 4. CHẨN ĐOÁN ==========
def _find_zone(xy, paths):
    if xy is None:
        return "UNCERTAIN"
    x, y = xy
    for zone, path in paths.items():
        if path.contains_point((x, y), radius=1e-6):
            return zone
    p = np.array([x, y])
    nearest = min(paths.keys(),
                  key=lambda z: np.min(np.linalg.norm(paths[z].vertices - p, axis=1)))
    return nearest

def duval_triangle_1(ch4, c2h4, c2h2):
    if any(np.isnan(x) or x < 0 for x in [ch4, c2h4, c2h2]):
        return "INVALID"
    total = ch4 + c2h4 + c2h2
    if total == 0:
        return "INVALID"
    pCH4  = ch4  / total * 100
    pC2H4 = c2h4 / total * 100
    pC2H2 = c2h2 / total * 100
    xy = ternary_to_xy(pCH4, pC2H2, pC2H4)
    return _find_zone(xy, PATHS_T1)

zone_t1 = duval_triangle_1(CH4, C2H4, C2H2)

fault_explanations = {
    "PD": "Partial Discharge",
    "T1": "Thermal fault < 300°C",
    "T2": "Thermal fault 300–700°C",
    "T3": "Thermal fault > 700°C",
    "D1": "Low energy electrical discharge",
    "D2": "High energy electrical discharge (arc)",
    "DT": "Mixed discharge + thermal fault",
    "UNCERTAIN": "Uncertain diagnosis",
    "INVALID": "Invalid gas data",
}

# ========== 5. TỌA ĐỘ ĐIỂM MẪU ==========
total1 = CH4 + C2H4 + C2H2
pCH4  = (CH4  / total1 * 100) if total1 > 0 else 0
pC2H4 = (C2H4 / total1 * 100) if total1 > 0 else 0
pC2H2 = (C2H2 / total1 * 100) if total1 > 0 else 0
sample_xy = ternary_to_xy(pCH4, pC2H2, pC2H4)

# ========== 6. MÀU SẮC ==========
zone_colors = {
    "PD": "#b3de69", "T1": "#80b1d3", "T2": "#fdb462",
    "T3": "#8dd3c7", "D1": "#ffffb3", "D2": "#bebada", "DT": "#fb8072",
}
zone_short_labels = {"PD": "PD", "T1": "T1", "T2": "T2", "T3": "T3", "D1": "D1", "D2": "D2", "DT": "DT"}

def text_color_for_bg(hex_color):
    hex_color = hex_color.lstrip('#')
    r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
    luminance = (0.299 * r + 0.587 * g + 0.114 * b) / 255
    return 'white' if luminance < 0.5 else 'black'

# ========== 7. VẼ ==========
fig, ax = plt.subplots(figsize=(9, 7))

# Vẽ vùng lỗi
for zone, path in PATHS_T1.items():
    color = zone_colors.get(zone, "gray")
    patch = PathPatch(path, facecolor=color, edgecolor='none', alpha=0.85)
    ax.add_patch(patch)
    centroid = path.vertices.mean(axis=0)
    ax.text(centroid[0], centroid[1], zone_short_labels.get(zone, zone),
            ha='center', va='center', fontweight='bold',
            color=text_color_for_bg(color), fontsize=10)

# Legend
legend_handles = [plt.Rectangle((0,0),1,1, facecolor=zone_colors[z], alpha=0.85,
                                label=f"{z}: {fault_explanations[z]}") for z in zone_colors]
ax.legend(handles=legend_handles, loc='center left', bbox_to_anchor=(1.02, 0.5),
          fontsize=8, framealpha=0.9, title="Fault Zones")

# ── Lưới % ──
tick_vals = [20, 40, 60, 80]
for tv in tick_vals:
    k = tv / 100
    y = h * k
    ax.plot([0.5*k, 1-0.5*k], [y, y], color='gray', lw=0.5, ls='--', alpha=0.6)
    ax.plot([0.5-0.5*k, 1-k], [h*(1-k), 0], color='gray', lw=0.5, ls='--', alpha=0.6)
    ax.plot([0.5+0.5*k, k], [h*(1-k), 0], color='gray', lw=0.5, ls='--', alpha=0.6)

# ── TICK % dọc theo cạnh ──
for tv in tick_vals:
    k = tv / 100
    # Cạnh trái (CH4 tăng từ dưới lên)
    ax.text(0.5*k - 0.03, h*k, f'{tv}%', ha='right', va='center', fontsize=7, color='#555555')
    # Cạnh phải (C2H4 tăng từ trên xuống)
    ax.text(0.5+0.5*k + 0.03, h*(1-k), f'{tv}%', ha='left', va='center', fontsize=7, color='#555555')
    # Cạnh đáy (C2H2 tăng từ phải qua trái)
    ax.text(1-k, -0.02, f'{tv}%', ha='center', va='top', fontsize=7, color='#555555')

# ── MŨI TÊN & NHÃN NGOÀI TAM GIÁC ──
arrow_style = "Simple, tail_width=0.5, head_width=4, head_length=4"
offset_arrow = 0.08
offset_label = 0.25   # tăng lên để nhãn hẳn ra ngoài
arrow_length = 0.18

def get_normal(p1, p2):
    """Trả về pháp tuyến ngoài (đơn vị) cho cạnh từ p1->p2."""
    if (np.allclose(p1, [0,0]) and np.allclose(p2, [0.5, h])) or \
       (np.allclose(p1, [0.5, h]) and np.allclose(p2, [0,0])):
        # cạnh trái
        n = np.array([-h, 0.5])
    elif (np.allclose(p1, [0.5, h]) and np.allclose(p2, [1,0])) or \
         (np.allclose(p1, [1,0]) and np.allclose(p2, [0.5, h])):
        # cạnh phải
        n = np.array([h, 0.5])
    elif (np.allclose(p1, [0,0]) and np.allclose(p2, [1,0])) or \
         (np.allclose(p1, [1,0]) and np.allclose(p2, [0,0])):
        # cạnh đáy
        n = np.array([0, -1])
    else:
        # fallback (không dùng)
        n = np.array([0, 1])
    return n / np.linalg.norm(n)

def draw_edge_arrow(ax, p1, p2, direction, offset_arrow, offset_label, label, color='black'):
    v = np.array([p2[0]-p1[0], p2[1]-p1[1]])
    v_norm = np.linalg.norm(v)
    if v_norm == 0:
        return
    uv = v / v_norm
    n = get_normal(p1, p2)   # pháp tuyến ngoài đơn vị
    mid = (np.array(p1) + np.array(p2)) / 2.0
    center_arrow = mid + n * offset_arrow
    start = center_arrow - uv * (arrow_length/2) * direction
    end   = center_arrow + uv * (arrow_length/2) * direction
    arrow = FancyArrowPatch(start, end, arrowstyle=arrow_style, color=color, lw=1.5)
    ax.add_patch(arrow)
    label_pos = mid + n * offset_label
    ax.text(label_pos[0], label_pos[1], label, fontsize=11, fontweight='bold',
            ha='center', va='center')

# Cạnh trái: CH4, tăng từ dưới lên → mũi tên từ (0,0) lên (0.5,h)
draw_edge_arrow(ax, [0,0], [0.5, h], direction=1, offset_arrow=offset_arrow,
                offset_label=offset_label, label='CH₄')
# Cạnh phải: C2H4, tăng từ trên xuống → mũi tên từ (0.5,h) xuống (1,0)
draw_edge_arrow(ax, [0.5, h], [1,0], direction=1, offset_arrow=offset_arrow,
                offset_label=offset_label, label='C₂H₄')
# Cạnh đáy: C2H2, tăng từ phải qua trái → mũi tên từ (1,0) sang (0,0)
draw_edge_arrow(ax, [0,0], [1,0], direction=-1, offset_arrow=offset_arrow,
                offset_label=offset_label, label='C₂H₂')

# ── NHÃN GÓC ──
ax.text(0.5, h + 0.02, 'CH₄', ha='center', va='bottom', fontsize=12, fontweight='bold')
ax.text(-0.05, -0.02, 'C₂H₂', ha='right', va='top', fontsize=12, fontweight='bold')
ax.text(1.05, -0.02, 'C₂H₄', ha='left', va='top', fontsize=12, fontweight='bold')

# Điểm mẫu
if sample_xy:
    ax.scatter(*sample_xy, color='red', marker='X', s=140, zorder=10,
               edgecolors='black', linewidths=1.5)
    ax.annotate(f'CH₄ {pCH4:.1f}%\nC₂H₂ {pC2H2:.1f}%\nC₂H₄ {pC2H4:.1f}%',
                xy=sample_xy, xytext=(sample_xy[0]+0.12, sample_xy[1]+0.08),
                fontsize=8, color='darkred', fontweight='bold',
                arrowprops=dict(arrowstyle='->', color='darkred'),
                bbox=dict(boxstyle='round', facecolor='white', edgecolor='darkred', alpha=0.9))

ax.set_xlim(-0.2, 1.45)
ax.set_ylim(-0.15, h + 0.18)
ax.set_aspect('equal')
ax.axis('off')
ax.set_title(f"Duval Triangle 1 – Fault: {zone_t1}\n{fault_explanations.get(zone_t1, '')}",
             fontsize=13, fontweight='bold', pad=12)

plt.tight_layout()
plt.show()