# dga/duval_triangle.py
from __future__ import annotations
import logging
import numpy as np, pandas as pd
from matplotlib.path import Path
logger=logging.getLogger(__name__)
SQRT3=np.sqrt(3.0)
TRIANGLE_HEIGHT=SQRT3/2.0
def ternary_to_xy(ch4,c2h2,c2h4):
    values=np.asarray([ch4,c2h2,c2h4],dtype=float)
    if not np.all(np.isfinite(values)):return None
    if np.any(values<0):return None
    total=float(values.sum())
    if total<=0:return None
    ch4_n=ch4/total;c2h2_n=c2h2/total;c2h4_n=c2h4/total
    x=c2h4_n+0.5*ch4_n
    y=TRIANGLE_HEIGHT*ch4_n
    return float(x),float(y)
def percentages(ch4,c2h4,c2h2):
    values=np.asarray([ch4,c2h4,c2h2],dtype=float)
    if not np.all(np.isfinite(values)):return None
    if np.any(values<0):return None
    total=float(values.sum())
    if total<=0:return None
    return(float(100.0*ch4/total),float(100.0*c2h4/total),float(100.0*c2h2/total))
def build_polygon_from_verts(coords):
    verts=np.asarray(coords,dtype=float)
    if len(verts)<3:raise ValueError("A polygon requires at least three vertices.")
    if not np.allclose(verts[0],verts[-1]):verts=np.vstack([verts,verts[0]])
    codes=[Path.MOVETO]+[Path.LINETO]*(len(verts)-2)+[Path.CLOSEPOLY]
    return Path(verts,codes)
REGION_COORDS={
    "PD":{"ch4":[98,100,98],"c2h2":[0,0,2],"c2h4":[2,0,0]},
    "D1":{"ch4":[0,0,64,87],"c2h2":[100,77,13,13],"c2h4":[0,23,23,0]},
    "D2":{"ch4":[0,0,31,47,64],"c2h2":[77,29,29,13,13],"c2h4":[23,71,40,40,23]},
    "DT":{"ch4":[0,0,35,46,96,87,47,31],"c2h2":[29,15,15,4,4,13,13,29],"c2h4":[71,85,50,50,0,0,40,40]},
    "T1":{"ch4":[76,80,98,98,96],"c2h2":[4,0,0,2,4],"c2h4":[20,20,2,0,0]},
    "T2":{"ch4":[46,50,80,76],"c2h2":[4,0,0,4],"c2h4":[50,50,20,20]},
    "T3":{"ch4":[0,0,50,35],"c2h2":[15,0,0,15],"c2h4":[85,100,50,50]},
}
def _build_triangle_paths():
    paths={}
    for zone,coords in REGION_COORDS.items():
        vertices=[]
        for ch4,c2h2,c2h4 in zip(coords["ch4"],coords["c2h2"],coords["c2h4"]):
            xy=ternary_to_xy(ch4,c2h2,c2h4)
            if xy is not None:vertices.append(xy)
        if len(vertices)>=3:paths[zone]=build_polygon_from_verts(vertices)
    return paths
PATHS_T1=_build_triangle_paths()
ZONE_SHORT_LABELS={"PD":"PD","T1":"T1","T2":"T2","T3":"T3","D1":"D1","D2":"D2","DT":"DT"}
def _safe_gas(value):
    try:value=float(value)
    except (TypeError,ValueError):return np.nan
    if not np.isfinite(value):return np.nan
    if value<0:return np.nan
    return value
def duval_triangle_1(ch4,c2h4,c2h2):
    ch4=_safe_gas(ch4);c2h4=_safe_gas(c2h4);c2h2=_safe_gas(c2h2)
    values=np.asarray([ch4,c2h4,c2h2],dtype=float)
    if not np.all(np.isfinite(values)):return "ABSTAIN"
    if np.any(values<0):return "ABSTAIN"
    total=float(values.sum())
    if total<0.1:return "ABSTAIN"
    xy=ternary_to_xy(ch4,c2h2,c2h4)
    if xy is None:return "ABSTAIN"
    zone_order=["PD","T1","T2","T3","DT","D1","D2"]
    for zone in zone_order:
        path=PATHS_T1.get(zone)
        if path is None:continue
        if path.contains_point(xy,radius=1e-10):return zone
    return "ABSTAIN"
def apply_duval_triangle(df):
    df=df.copy()
    xs=[];ys=[];ch4_percent=[];c2h4_percent=[];c2h2_percent=[];faults=[]
    for _,row in df.iterrows():
        ch4=_safe_gas(row.get("ch4",np.nan));c2h4=_safe_gas(row.get("c2h4",np.nan));c2h2=_safe_gas(row.get("c2h2",np.nan))
        values=np.asarray([ch4,c2h4,c2h2],dtype=float)
        if not np.all(np.isfinite(values)):
            xs.append(np.nan);ys.append(np.nan);ch4_percent.append(np.nan);c2h4_percent.append(np.nan);c2h2_percent.append(np.nan);faults.append("ABSTAIN");continue
        total=float(values.sum())
        if total<0.1:
            xs.append(np.nan);ys.append(np.nan);ch4_percent.append(np.nan);c2h4_percent.append(np.nan);c2h2_percent.append(np.nan);faults.append("ABSTAIN");continue
        xy=ternary_to_xy(ch4,c2h2,c2h4);pct=percentages(ch4,c2h4,c2h2)
        if xy is None or pct is None:
            xs.append(np.nan);ys.append(np.nan);ch4_percent.append(np.nan);c2h4_percent.append(np.nan);c2h2_percent.append(np.nan);faults.append("ABSTAIN");continue
        xs.append(xy[0]);ys.append(xy[1]);ch4_percent.append(pct[0]);c2h4_percent.append(pct[1]);c2h2_percent.append(pct[2]);fault=duval_triangle_1(ch4,c2h4,c2h2);faults.append(fault)
    df["t_x"]=xs;df["t_y"]=ys;df["duval_ch4_pct"]=ch4_percent;df["duval_c2h4_pct"]=c2h4_percent;df["duval_c2h2_pct"]=c2h2_percent;df["duval_triangle_fault"]=faults;logger.debug("Duval Triangle 1 diagnostic applied.")
    return df