# keygas.py
from __future__ import annotations

import numpy as np
import pandas as pd

# IEEE C57.104 Key Gas Method
#
# Output labels:
#   Normal
#   PD
#   D2
#   THERMAL_OIL
#   THERMAL_CELLULOSE
#   UNCERTAIN

MIN_TDCG = 100.0


def _gas(row: pd.Series, name: str) -> float:
    value = row.get(name.lower(), row.get(name.upper(), np.nan))

    if pd.isna(value):
        return 0.0

    return float(max(value, 0.0))


def key_gas_method(row: pd.Series) -> str:

    h2 = _gas(row, "H2")
    ch4 = _gas(row, "CH4")
    c2h6 = _gas(row, "C2H6")
    c2h4 = _gas(row, "C2H4")
    c2h2 = _gas(row, "C2H2")
    co = _gas(row, "CO")

    tdcg = row.get("tdcg", np.nan)

    if pd.isna(tdcg):
        tdcg = (
            h2
            + ch4
            + c2h6
            + c2h4
            + c2h2
            + co
        )

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

    dominant = max(gases, key=gases.get)

    # IEEE Key Gas mapping

    if dominant == "H2":

        # H2 only -> PD
        # H2 + C2H2 -> Arcing
        if c2h2 > 0:
            return "D2"

        return "PD"

    if dominant == "C2H2":
        return "D2"

    if dominant == "C2H4":
        return "THERMAL_OIL"

    if dominant == "CO":
        return "THERMAL_CELLULOSE"

    # CH4 hoặc C2H6 không phải key gas
    return "UNCERTAIN"


def apply_key_gas(df: pd.DataFrame) -> pd.DataFrame:

    df = df.copy()

    df["keygas_fault"] = df.apply(
        key_gas_method,
        axis=1,
    )

    return df