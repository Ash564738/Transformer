# dga/rogers.py
from __future__ import annotations
from dataclasses import field
import logging
from typing import List
import numpy as np, pandas as pd
from config import config as cfg
logger = logging.getLogger(__name__)
ROGERS_R1_LOW=0.1;ROGERS_R1_HIGH=3.0;ROGERS_R2_LOW=0.1;ROGERS_R2_HIGH=1.0;ROGERS_R3_LOW=1.0;ROGERS_R3_HIGH=3.0
DIAGNOSTIC_GASES=["h2","ch4","c2h2","c2h4","c2h6"]
L1_LIMITS={"h2":100,"ch4":120,"c2h2":1,"c2h4":50,"c2h6":65}
def _gas(row,name):
    value=row.get(name,np.nan)
    try:value=float(value)
    except (TypeError,ValueError):return np.nan
    return value if np.isfinite(value) and value>=0 else np.nan
def _safe_ratio(num,den):
    if pd.isna(num) or pd.isna(den):return np.nan
    if den==0:return np.inf if num>0 else np.nan
    return float(num)/float(den)
def _rogers_applicable(row):
    for gas in DIAGNOSTIC_GASES:
        value=_gas(row,gas)
        if np.isfinite(value) and value>=L1_LIMITS[gas]:return True
    return False
def _code_r1(ratio):
    if not np.isfinite(ratio):return None
    return 0 if ratio<ROGERS_R1_LOW else 1 if ratio<=ROGERS_R1_HIGH else 2
def _code_r2(ratio):
    if not np.isfinite(ratio):return None
    return 0 if ratio<ROGERS_R2_LOW else 1 if ratio<=ROGERS_R2_HIGH else 2
def _code_r3(ratio):
    if not np.isfinite(ratio):return None
    return 0 if ratio<ROGERS_R3_LOW else 1 if ratio<=ROGERS_R3_HIGH else 2
def rogers_ratio_method(row):
    if not _rogers_applicable(row):return "ABSTAIN"
    c2h2=_gas(row,"c2h2");c2h4=_gas(row,"c2h4");ch4=_gas(row,"ch4");h2=_gas(row,"h2");c2h6=_gas(row,"c2h6")
    values=[c2h2,c2h4,ch4,h2,c2h6]
    if not all(np.isfinite(v) for v in values):return "ABSTAIN"
    r1=_safe_ratio(c2h2,c2h4);r2=_safe_ratio(ch4,h2);r3=_safe_ratio(c2h4,c2h6)
    if not all(np.isfinite(v) for v in [r1,r2,r3]):return "ABSTAIN"
    code=(_code_r1(r1),_code_r2(r2),_code_r3(r3))
    if any(v is None for v in code):return "ABSTAIN"
    mapping={(0,1,0):"NORMAL",(0,0,0):"PD",(1,1,2):"D2",(0,1,1):"T1",(0,2,1):"T2",(0,2,2):"T3"}
    return mapping.get(code,"ABSTAIN")
def apply_rogers(df):
    df=df.copy()
    df["r1_c2h2_c2h4"]=df.apply(lambda row:_safe_ratio(_gas(row,"c2h2"),_gas(row,"c2h4")),axis=1)
    df["r2_ch4_h2"]=df.apply(lambda row:_safe_ratio(_gas(row,"ch4"),_gas(row,"h2")),axis=1)
    df["r3_c2h4_c2h6"]=df.apply(lambda row:_safe_ratio(_gas(row,"c2h4"),_gas(row,"c2h6")),axis=1)
    df["rogers_fault"]=df.apply(rogers_ratio_method,axis=1)
    logger.debug("Rogers diagnostic applied.")
    return df