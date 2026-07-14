# keygas.py
from __future__ import annotations

import numpy as np
import pandas as pd

# IEEE C57.104 Key Gas interpretation
# Fault labels:
# Normal
# PD
# D2
# T1
# T2
# T3
# Cellulose
# Mixed
# Uncertain

MIN_TDCG = 100.0


def _gas(row: pd.Series, name: str) -> float:
    v = row.get(name.lower(), row.get(name.upper(), np.nan))
    if pd.isna(v):
        return 0.0
    return float(max(v, 0.0))


def key_gas_method(row: pd.Series) -> str:
    h2 = _gas(row, "H2")
    ch4 = _gas(row, "CH4")
    c2h6 = _gas(row, "C2H6")
    c2h4 = _gas(row, "C2H4")
    c2h2 = _gas(row, "C2H2")
    co = _gas(row, "CO")
    co2 = _gas(row, "CO2")

    tdcg = row.get("tdcg", np.nan)
    if pd.isna(tdcg):
        tdcg = h2 + ch4 + c2h6 + c2h4 + c2h2 + co

    if tdcg < MIN_TDCG:
        return "Normal"

    gases = {
        "H2": h2,
        "CH4": ch4,
        "C2H6": c2h6,
        "C2H4": c2h4,
        "C2H2": c2h2,
        "CO": co,
    }

    total = sum(gases.values())

    if total <= 0:
        return "Uncertain"

    pct = {k: 100.0 * v / total for k, v in gases.items()}
    dominant = max(gases, key=gases.get)

    # ------------------------------------------------------------------
    # IEEE Key Gas
    # ------------------------------------------------------------------

    # Cellulose degradation
    ratio_co2_co = np.nan
    if co > 0:
        ratio_co2_co = co2 / co

    if dominant == "CO":
        if np.isnan(ratio_co2_co) or ratio_co2_co < 7:
            return "Cellulose"

    # Partial discharge
    if dominant == "H2":
        if pct["H2"] >= 50 and ch4 <= 0.3 * h2:
            return "PD"

    # High-energy discharge / arcing
    if dominant == "C2H2":
        return "D2"

    if c2h2 >= 0.3 * tdcg and h2 >= 0.2 * tdcg:
        return "D2"

    # Thermal faults
    if dominant == "C2H4":
        if c2h2 > 0.10 * c2h4:
            return "T3"

        if c2h4 >= 2.0 * ch4:
            return "T3"

        if c2h4 >= ch4:
            return "T2"

    if dominant in ("CH4", "C2H6"):
        return "T1"

    # Mixed electrical + thermal
    electrical = h2 + c2h2
    thermal = ch4 + c2h6 + c2h4

    if electrical > 0 and thermal > 0:
        if electrical / total > 0.30 and thermal / total > 0.30:
            return "Mixed"

    return "Uncertain"


def apply_key_gas(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["keygas_fault"] = df.apply(key_gas_method, axis=1)
    return df