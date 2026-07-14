# consensus.py

from typing import Dict
import pandas as pd

from . import (
    keygas,
    iec60599,
    rogers,
    duval_triangle,
    duval_pentagon
)



# ==========================================================
# Fault groups
# ==========================================================

FAULT_GROUP = {
    "PD": "electrical",
    "D1": "electrical",
    "D2": "electrical",

    "T1": "thermal",
    "T2": "thermal",
    "T3": "thermal",
    "T3-H": "thermal",
    "THERMAL": "thermal",
    "THERMAL_OIL": "thermal",

    "C": "cellulose",
    "Cellulose": "cellulose",
    "THERMAL_CELLULOSE": "cellulose",

    "NORMAL": "normal",
}

# ==========================================================
# Normalize fault label
# ==========================================================

def normalize_fault(label):
    if label is None:
        return "UNCERTAIN"

    label = str(label).strip()

    # Chỉ chuyển về UNCERTAIN với các trường hợp thực sự không xác định
    invalid = {
        "",
        "UNCERTAIN",
    }

    if label in invalid:
        return "UNCERTAIN"

    return label



# ==========================================================
# Confidence score
# ==========================================================

def confidence(votes: Dict[str,str]) -> float:

    valid = [
        normalize_fault(v)
        for v in votes.values()
        if normalize_fault(v) != "INVALID_LOW_GAS"
    ]


    if len(valid) == 0:
        return 0.0



    count = pd.Series(valid).value_counts()


    return round(

        count.iloc[0]
        /
        len(valid)
        *
        100,

        1

    )



# ==========================================================
# Aggregate diagnosis
# ==========================================================

def aggregate_votes(votes: Dict[str,str]):


    cleaned = {

        k: normalize_fault(v)

        for k,v in votes.items()

    }



    valid = [
        v
        for v in cleaned.values()
        if v != "INVALID_LOW_GAS"
    ]


    if not valid:
        return "UNCERTAIN"



    count = pd.Series(valid).value_counts()



    # Majority vote

    if count.iloc[0] >= 2:

        return count.index[0]



    # Fault family conflict

    groups=set()


    for fault in valid:

        group = FAULT_GROUP.get(fault)


        if group:
            groups.add(group)



    if len(groups) > 1:

        if "normal" not in groups:

            return "MIXED"



    # Priority

    priority = [

        "duval_pentagon_fault",

        "duval_triangle_fault",

        "iec_fault",

        "rogers_fault",

        "keygas_fault"

    ]


    for p in priority:

        fault = cleaned.get(p)


        if fault != "UNCERTAIN":

            return fault



    return "UNCERTAIN"





# ==========================================================
# Apply all DGA methods
# ==========================================================

def apply_consensus(df):


    df = keygas.apply_key_gas(df)

    df = iec60599.apply_iec(df)

    df = rogers.apply_rogers(df)

    df = duval_triangle.apply_duval_triangle(df)

    df = duval_pentagon.apply_duval_pentagon(
        df,
        pentagon="P2"
    )



    def make_votes(row):

        return {

            "keygas_fault":
                row.get(
                    "keygas_fault",
                    "UNCERTAIN"
                ),


            "iec_fault":
                row.get(
                    "iec_fault",
                    "UNCERTAIN"
                ),


            "rogers_fault":
                row.get(
                    "rogers_fault",
                    "UNCERTAIN"
                ),


            "duval_triangle_fault":
                row.get(
                    "duval_triangle_fault",
                    "UNCERTAIN"
                ),


            "duval_pentagon_fault":
                row.get(
                    "duval_pentagon_fault",
                    "UNCERTAIN"
                )

        }



    # lưu vote nếu cần debug

    df["diagnostic_votes"] = df.apply(
        lambda r:
            make_votes(r),
        axis=1
    )



    df["consensus_fault"] = df.apply(

        lambda r:

            aggregate_votes(
                make_votes(r)
            ),

        axis=1

    )



    df["diagnostic_confidence"] = df.apply(

        lambda r:

            confidence(
                make_votes(r)
            ),

        axis=1

    )



    return df