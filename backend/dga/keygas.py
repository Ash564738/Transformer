# dga/keygas.py
from __future__ import annotations
import logging
import numpy as np, pandas as pd
logger = logging.getLogger(__name__)

def _gas(row, name):
    value = row.get(name, np.nan)
    try: value = float(value)
    except (TypeError, ValueError): return np.nan
    return value if np.isfinite(value) and value >= 0 else np.nan

def _safe_tdcg(row):
    value = row.get("tdcg", np.nan)
    try: value = float(value)
    except (TypeError, ValueError): value = np.nan
    if np.isfinite(value) and value >= 0: return value
    gas_names = ["h2","ch4","c2h6","c2h4","c2h2","co"]
    values = [_gas(row, g) for g in gas_names]
    if not all(np.isfinite(v) for v in values): return np.nan
    return float(sum(values))

def _all_required_gases_present(row):
    required = ["h2","ch4","c2h6","c2h4","c2h2","co"]
    return all(np.isfinite(_gas(row, g)) for g in required)

def key_gas_method(row):
    if not _all_required_gases_present(row): return "ABSTAIN"
    h2 = _gas(row,"h2"); ch4 = _gas(row,"ch4"); c2h6 = _gas(row,"c2h6")
    c2h4 = _gas(row,"c2h4"); c2h2 = _gas(row,"c2h2"); co = _gas(row,"co")
    combustible = {"h2":h2,"ch4":ch4,"c2h6":c2h6,"c2h4":c2h4,"c2h2":c2h2,"co":co}
    total = float(sum(combustible.values()))
    if total <= 0: return "ABSTAIN"
    proportions = {g: v/total for g,v in combustible.items()}
    if c2h2 > 0 and h2 > 0:
        if proportions["c2h2"] >= 0.05 and proportions["h2"] >= 0.20: return "D2"
    hydrocarbon_total = ch4 + c2h6 + c2h4 + c2h2
    if h2 > 0:
        h2_fraction = proportions["h2"]
        if h2_fraction >= 0.60 and h2 >= hydrocarbon_total:
            if c2h2 <= 0.10*h2: return "PD"
    hydrocarbon_plus_h2 = h2 + ch4 + c2h6 + c2h4 + c2h2
    if co > 0:
        co_fraction = proportions["co"]
        if co_fraction >= 0.60 and co >= hydrocarbon_plus_h2: return "THERMAL_CELLULOSE"
    thermal_companions = c2h6 + ch4 + h2
    if c2h4 > 0:
        c2h4_fraction = proportions["c2h4"]
        if c2h4_fraction >= 0.50 and c2h4 >= thermal_companions and c2h2 <= 0.10*c2h4: return "THERMAL_OIL"
    return "ABSTAIN"

def apply_key_gas(df):
    df = df.copy()
    df["keygas_fault"] = df.apply(key_gas_method, axis=1)
    if "tdcg" not in df.columns: df["tdcg"] = df.apply(_safe_tdcg, axis=1)
    logger.debug("IEEE C57.104-2019 Key Gas diagnostic applied.")
    if logger.isEnabledFor(logging.DEBUG):
        sample_cols = ["h2","ch4","c2h6","c2h4","c2h2","co","co2","tdcg","keygas_fault"]
        available_cols = [col for col in sample_cols if col in df.columns]
        logger.debug("Sample Key Gas results:\n%s", df[available_cols].head(5).to_string())
    return df