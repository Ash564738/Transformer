# dga/duval_pentagon.py
from __future__ import annotations
import logging
from typing import Dict, Iterable, Optional, Sequence, Tuple
import numpy as np, pandas as pd
from matplotlib.path import Path
logger = logging.getLogger(__name__)

PENTAGON_GASES: Tuple[str, ...] = ("h2", "c2h6", "ch4", "c2h4", "c2h2")
PENTAGON_VERTICES_RAW: Dict[str, np.ndarray] = {"H2": np.array([0.0, 40.0]), "C2H6": np.array([-38.0, 12.4]), "CH4": np.array([-23.5, -32.4]), "C2H4": np.array([23.5, -32.4]), "C2H2": np.array([38.0, 12.4])}
PENTAGON_VERTICES = PENTAGON_VERTICES_RAW

def _zone(points: Sequence[Tuple[float, float]]) -> Tuple[Tuple[float, float], ...]:
    return tuple((float(x), float(y)) for x, y in points)

PENTAGON_1_ZONES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "PD": _zone([(0.0, 33.0), (-1.0, 33.0), (-1.0, 24.5), (0.0, 24.5)]),
    "D1": _zone([(0.0, 40.0), (38.0, 12.0), (32.0, -6.1), (4.0, 16.0), (0.0, 1.5)]),
    "D2": _zone([(4.0, 16.0), (32.0, -6.1), (24.3, -30.0), (0.0, -3.0), (0.0, 1.5)]),
    "T3": _zone([(0.0, -3.0), (24.3, -30.0), (23.5, -32.4), (1.0, -32.4), (-6.0, -4.0)]),
    "T2": _zone([(-6.0, -4.0), (1.0, -32.4), (-22.5, -32.4)]),
    "T1": _zone([(-6.0, -4.0), (-22.5, -32.4), (-23.5, -32.4), (-35.0, 3.0), (0.0, 1.5), (0.0, -3.0)]),
    "S": _zone([(0.0, 1.5), (-35.0, 3.1), (-38.0, 12.4), (0.0, 40.0), (0.0, 33.0), (-1.0, 33.0), (-1.0, 24.5), (0.0, 24.5)]),
}
PENTAGON_2_ZONES: Dict[str, Tuple[Tuple[float, float], ...]] = {
    "PD": PENTAGON_1_ZONES["PD"],
    "D1": PENTAGON_1_ZONES["D1"],
    "D2": PENTAGON_1_ZONES["D2"],
    "S": PENTAGON_1_ZONES["S"],
    "T3_H": _zone([(0.0, -3.0), (24.3, -30.0), (23.5, -32.4), (2.5, -32.4), (-3.5, -3.0)]),
    "C": _zone([(-3.5, -3.0), (2.5, -32.4), (-21.5, -32.4), (-11.0, -8.0)]),
    "O": _zone([(-3.5, -3.0), (-11.0, -8.0), (-21.5, -32.4), (-23.5, -32.4), (-35.0, 3.1), (0.0, 1.5), (0.0, -3.0)]),
}
ZONE_EXPLANATIONS: Dict[str, str] = {"PD": "Partial discharge", "D1": "Low-energy electrical discharge", "D2": "High-energy electrical discharge", "T1": "Low-temperature thermal fault", "T2": "Intermediate-temperature thermal fault", "T3": "High-temperature thermal fault", "T3_H": "High-temperature thermal fault, hydrogen-associated region", "C": "Thermal fault involving paper/cellulose carbonization", "O": "Oil-related thermal fault region", "S": "Stray gassing"}
ZONE_COLORS: Dict[str, str] = {"PD": "#cfff7c", "D1": "#ffffb3", "D2": "#cec9ec", "T1": "#9dd6ff", "T2": "#ffbd72", "T3": "#9de4d9", "T3_H": "#90ccc2", "C": "#fb8072", "O": "#ffd3ea", "S": "#ffcfab"}
MIN_PENTAGON_TOTAL = 0.1

def _safe_gas(value) -> float:
    try: value = float(value)
    except (TypeError, ValueError): return np.nan
    if not np.isfinite(value): return np.nan
    if value < 0.0: return np.nan
    return value

def _read_pentagon_gases(row: pd.Series) -> Optional[Tuple[float, ...]]:
    values = tuple(_safe_gas(row.get(gas, np.nan)) for gas in PENTAGON_GASES)
    if not all(np.isfinite(v) for v in values): return None
    return values

def _polygon_centroid(points: Iterable[Tuple[float, float]]) -> Optional[Tuple[float, float]]:
    pts = np.asarray(list(points), dtype=float)
    if pts.ndim != 2: return None
    if pts.shape[0] < 3 or pts.shape[1] != 2: return None
    if not np.all(np.isfinite(pts)): return None
    x = pts[:, 0]; y = pts[:, 1]
    x_next = np.roll(x, -1); y_next = np.roll(y, -1)
    cross = x * y_next - x_next * y
    signed_area = 0.5 * np.sum(cross)
    if not np.isfinite(signed_area) or abs(signed_area) <= 1e-12: return None
    cx = np.sum((x + x_next) * cross) / (6.0 * signed_area)
    cy = np.sum((y + y_next) * cross) / (6.0 * signed_area)
    if not (np.isfinite(cx) and np.isfinite(cy)): return None
    return float(cx), float(cy)

def duval_pentagon_centroid(h2, ch4, c2h6, c2h4, c2h2) -> Optional[Tuple[float, float]]:
    values = {"H2": _safe_gas(h2), "C2H6": _safe_gas(c2h6), "CH4": _safe_gas(ch4), "C2H4": _safe_gas(c2h4), "C2H2": _safe_gas(c2h2)}
    if not all(np.isfinite(v) for v in values.values()): return None
    total = float(sum(values.values()))
    if not np.isfinite(total) or total <= 0.0 or total < MIN_PENTAGON_TOTAL: return None
    fractions = {gas: value / total for gas, value in values.items()}
    points = []
    for gas in ("H2", "C2H6", "CH4", "C2H4", "C2H2"):
        vertex = PENTAGON_VERTICES_RAW[gas]; fraction = fractions[gas]
        points.append((float(vertex[0] * fraction), float(vertex[1] * fraction)))
    return _polygon_centroid(points)

def _point_on_segment(point, a, b, tolerance=1e-8) -> bool:
    px, py = point; ax, ay = a; bx, by = b
    abx = bx - ax; aby = by - ay
    apx = px - ax; apy = py - ay
    cross = abx * apy - aby * apx
    if abs(cross) > tolerance: return False
    dot = apx * abx + apy * aby
    if dot < -tolerance: return False
    ab_squared = abx * abx + aby * aby
    if dot > ab_squared + tolerance: return False
    return True

def _point_on_polygon_boundary(point, polygon, tolerance=1e-8) -> bool:
    n = len(polygon)
    for i in range(n):
        a = polygon[i]; b = polygon[(i + 1) % n]
        if _point_on_segment(point, a, b, tolerance=tolerance): return True
    return False

def _find_pentagon_zone(xy, zones, paths, boundary_tolerance=1e-8) -> str:
    if xy is None: return "ABSTAIN"
    x, y = xy
    if not (np.isfinite(x) and np.isfinite(y)): return "ABSTAIN"
    point = (float(x), float(y))
    boundary_hits = [zone for zone, polygon in zones.items() if _point_on_polygon_boundary(point, polygon, tolerance=boundary_tolerance)]
    if len(boundary_hits) > 1: return "ABSTAIN"
    if len(boundary_hits) == 1: return boundary_hits[0]
    hits = [zone for zone, path in paths.items() if path.contains_point(point, radius=0.0)]
    if len(hits) == 1: return hits[0]
    if len(hits) > 1: logger.warning("Duval Pentagon point %.10f, %.10f overlaps zones: %s", x, y, hits)
    return "ABSTAIN"

PATHS_P1: Dict[str, Path] = {name: Path(np.asarray(polygon, dtype=float)) for name, polygon in PENTAGON_1_ZONES.items()}
PATHS_P2: Dict[str, Path] = {name: Path(np.asarray(polygon, dtype=float)) for name, polygon in PENTAGON_2_ZONES.items()}

def classify_duval_pentagon(row: pd.Series) -> Tuple[Optional[Tuple[float, float]], str, str]:
    gases = _read_pentagon_gases(row)
    if gases is None: return None, "ABSTAIN", "ABSTAIN"
    h2, c2h6, ch4, c2h4, c2h2 = gases
    total = float(h2 + c2h6 + ch4 + c2h4 + c2h2)
    if not np.isfinite(total) or total <= 0.0 or total < MIN_PENTAGON_TOTAL: return None, "ABSTAIN", "ABSTAIN"
    xy = duval_pentagon_centroid(h2=h2, ch4=ch4, c2h6=c2h6, c2h4=c2h4, c2h2=c2h2)
    if xy is None: return None, "ABSTAIN", "ABSTAIN"
    fault_p1 = _find_pentagon_zone(xy, PENTAGON_1_ZONES, PATHS_P1)
    fault_p2 = _find_pentagon_zone(xy, PENTAGON_2_ZONES, PATHS_P2)
    return xy, fault_p1, fault_p2

def apply_duval_pentagon_dual(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    xs = []; ys = []; p1_faults = []; p2_faults = []
    for _, row in out.iterrows():
        xy, p1_fault, p2_fault = classify_duval_pentagon(row)
        if xy is None: xs.append(np.nan); ys.append(np.nan)
        else: xs.append(float(xy[0])); ys.append(float(xy[1]))
        p1_faults.append(p1_fault); p2_faults.append(p2_fault)
    out["p_x"] = xs; out["p_y"] = ys
    out["duval_pentagon_p1_fault"] = p1_faults
    out["duval_pentagon_p2_fault"] = p2_faults
    out["fault_p1"] = p1_faults
    out["fault_p2"] = p2_faults
    out["duval_pentagon_p1_active"] = (out["duval_pentagon_p1_fault"] != "ABSTAIN")
    out["duval_pentagon_p2_active"] = (out["duval_pentagon_p2_fault"] != "ABSTAIN")
    return out

def apply_duval_pentagon(df: pd.DataFrame, pentagon: str = "P2") -> pd.DataFrame:
    out = apply_duval_pentagon_dual(df)
    requested = str(pentagon).strip().upper()
    if requested == "P1": out["duval_pentagon_fault"] = out["duval_pentagon_p1_fault"]
    elif requested == "P2": out["duval_pentagon_fault"] = out["duval_pentagon_p2_fault"]
    else: raise ValueError(f"Unknown pentagon={pentagon!r}; expected 'P1' or 'P2'.")
    return out

def get_duval_pentagon_faults(row: pd.Series) -> Dict[str, str]:
    _, p1, p2 = classify_duval_pentagon(row)
    return {"p1": p1, "p2": p2}

def diagnose_duval_pentagon_row(row: pd.Series) -> Dict[str, object]:
    xy, p1, p2 = classify_duval_pentagon(row)
    result = {"p_x": np.nan, "p_y": np.nan, "p1_fault": p1, "p2_fault": p2,
              "p1_explanation": ZONE_EXPLANATIONS.get(p1, "No classification"),
              "p2_explanation": ZONE_EXPLANATIONS.get(p2, "No classification")}
    if xy is not None:
        result["p_x"] = float(xy[0]); result["p_y"] = float(xy[1])
    return result

def _self_test() -> None:
    row = pd.Series({"h2": 21.0, "c2h6": 159.0, "ch4": 0.0, "c2h4": 150.0, "c2h2": 505.0})
    result = diagnose_duval_pentagon_row(row)
    logger.info("Duval Pentagon self-test: %s", result)
    assert result["p1_fault"] == "D2", f"Expected P1=D2 for supplied test vector, got {result['p1_fault']}"
    assert result["p2_fault"] == "D2", f"Expected P2=D2 for supplied test vector, got {result['p2_fault']}"
    assert np.isfinite(result["p_x"])
    assert np.isfinite(result["p_y"])

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    _self_test()