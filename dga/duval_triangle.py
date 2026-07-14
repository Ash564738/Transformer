# duval_triangle.py

import numpy as np
import pandas as pd
from matplotlib.path import Path



# =====================================================
# Duval Triangle 1
#
# Vertex:
#
# CH4  = (0,0.866)
# C2H4 = (1,0)
# C2H2 = (0,0)
#
# =====================================================


def ternary_to_xy(
        ch4,
        c2h4,
        c2h2
):

    total = (
        ch4 +
        c2h4 +
        c2h2
    )

    if total <= 0:
        return None


    ch4 = ch4 / total
    c2h4 = c2h4 / total
    c2h2 = c2h2 / total


    #
    # Cartesian conversion
    #
    x = (
        c2h4
        +
        0.5 * ch4
    )


    y = (
        ch4
        *
        0.8660254037844386
    )


    return x,y



# =====================================================
# Region definition
#
# Define by ternary percentage:
#
# (CH4, C2H4, C2H2)
#
# =====================================================


REGIONS_PERCENT = {


    "PD":[
        (98,2,0),
        (98,0,2),
        (100,0,0)
    ],



    "D1":[
        (0,77,23),
        (64,13,23),
        (87,13,0),
        (100,0,0),
    ],



    "D2":[
        (64,13,23),
        (47,13,40),
        (0,29,71),
        (0,77,23),
    ],



    "DT":[
        (0,15,85),
        (35,15,50),
        (46,4,50),
        (96,4,0),
        (87,13,0),
        (47,13,40),
        (31,29,40),
        (0,29,71),
    ],



    "T1":[
        (96,4,0),
        (98,2,0),
        (98,0,2),
        (80,0,20),
        (76,4,20),
    ],



    "T2":[
        (76,4,20),
        (80,0,20),
        (50,0,50),
        (46,4,50),
    ],



    "T3":[
        (0,15,85),
        (35,15,50),
        (50,0,50),
        (0,0,100),
    ]

}




# =====================================================
# Convert region ternary to xy
# =====================================================


def build_polygon(points):

    xy=[]

    for ch4,c2h4,c2h2 in points:

        p=ternary_to_xy(
            ch4,
            c2h4,
            c2h2
        )

        xy.append(p)


    return Path(xy)



PATHS={

    name:
    build_polygon(poly)

    for name,poly
    in REGIONS_PERCENT.items()

}




# =====================================================
# Diagnosis
# =====================================================


def duval_triangle_1(
        ch4,
        c2h4,
        c2h2
):


    if any(
        pd.isna(x)
        for x in [
            ch4,
            c2h4,
            c2h2
        ]
    ):
        return "INVALID"



    if (
        ch4 < 0 or
        c2h4 < 0 or
        c2h2 < 0
    ):
        return "INVALID"



    xy = ternary_to_xy(
        ch4,
        c2h4,
        c2h2
    )


    if xy is None:
        return "INVALID_LOW_GAS"



    for zone,path in PATHS.items():

        if path.contains_point(
            xy
        ):
            return zone



    return "OUTSIDE"




# =====================================================
# Apply dataframe
# =====================================================


def apply_duval_triangle(
        df:pd.DataFrame
):

    df=df.copy()


    results=[]


    xs=[]
    ys=[]


    for _,row in df.iterrows():


        ch4=row.get(
            "CH4",
            np.nan
        )


        c2h4=row.get(
            "C2H4",
            np.nan
        )


        c2h2=row.get(
            "C2H2",
            np.nan
        )


        xy=ternary_to_xy(
            ch4,
            c2h4,
            c2h2
        )


        if xy:

            xs.append(
                xy[0]
            )

            ys.append(
                xy[1]
            )

        else:

            xs.append(
                np.nan
            )

            ys.append(
                np.nan
            )


        results.append(

            duval_triangle_1(
                ch4,
                c2h4,
                c2h2
            )

        )



    df["duval_triangle_x"]=xs

    df["duval_triangle_y"]=ys

    df["duval_triangle_fault"]=results


    return df