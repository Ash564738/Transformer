# duval_pentagon.py

import numpy as np
import pandas as pd


# ==========================================================
# Duval Pentagon Zones
# ==========================================================

PENTAGON_1_ZONES = {

    "PD": [
        (0,33),
        (-1,33),
        (-1,24.5),
        (0,24.5),
    ],

    "D1": [
        (0,40),
        (38,12),
        (32,-6.1),
        (4,16),
        (0,1.5),
    ],

    "D2": [
        (4,16),
        (32,-6.1),
        (24.3,-30),
        (0,-3),
        (0,1.5),
    ],

    "T3": [
        (0,-3),
        (24.3,-30),
        (23.5,-32.4),
        (1,-32),
        (-6,-4),
    ],

    "T2": [
        (-6,-4),
        (1,-32.4),
        (-22.5,-32.4),
    ],

    "T1": [
        (-6,-4),
        (-22.5,-32.4),
        (-23.5,-32.4),
        (-35,3),
        (0,1.5),
        (0,-3),
    ],

    "S": [
        (0,1.5),
        (-35,3.1),
        (-38,12.4),
        (0,40),
        (0,33),
        (-1,33),
        (-1,24.5),
        (0,24.5),
    ]
}


PENTAGON_2_ZONES = {

    "PD": PENTAGON_1_ZONES["PD"],
    "D1": PENTAGON_1_ZONES["D1"],
    "D2": PENTAGON_1_ZONES["D2"],

    "T3-H": [
        (0,-3),
        (24.3,-30),
        (23.5,-32.4),
        (2.5,-32.4),
        (-3.5,-3),
    ],

    "C": [
        (-3.5,-3),
        (2.5,-32.4),
        (-21.5,-32.4),
        (-11,-8),
    ],

    "O": [
        (-3.5,-3),
        (-11,-8),
        (-21.5,-32.4),
        (-23.5,-32.4),
        (-35,3.1),
        (0,1.5),
        (0,-3),
    ],

    "S": PENTAGON_1_ZONES["S"]
}



# ==========================================================
# Convert 5 gases -> centroid
# ==========================================================

def duval_pentagon_centroid(
        h2,
        ch4,
        c2h6,
        c2h4,
        c2h2
):

    total = (
        h2 +
        ch4 +
        c2h6 +
        c2h4 +
        c2h2
    )

    if total <= 0:
        return None


    # Percentage
    pct = {

        "H2":
            h2 / total,

        "C2H6":
            c2h6 / total,

        "CH4":
            ch4 / total,

        "C2H4":
            c2h4 / total,

        "C2H2":
            c2h2 / total
    }


    #
    # Duval Pentagon 40% vertices
    #
    vertices = [

        pct["H2"] *
        np.array([0,40]),

        pct["C2H6"] *
        np.array([-38,12.4]),

        pct["CH4"] *
        np.array([-23.5,-32.4]),

        pct["C2H4"] *
        np.array([23.5,-32.4]),

        pct["C2H2"] *
        np.array([38,12.4]),
    ]


    # Polygon centroid
    A = 0
    Cx = 0
    Cy = 0


    n = len(vertices)


    for i in range(n):

        x1,y1 = vertices[i]

        x2,y2 = vertices[
            (i+1)%n
        ]


        cross = (
            x1*y2 -
            x2*y1
        )


        A += cross


        Cx += (
            x1+x2
        ) * cross


        Cy += (
            y1+y2
        ) * cross


    A *= 0.5


    if abs(A) < 1e-9:
        return None


    Cx /= (6*A)

    Cy /= (6*A)


    return Cx,Cy



# ==========================================================
# Point inside polygon
# ==========================================================

def point_inside_polygon(
        x,
        y,
        polygon
):

    inside=False

    j=len(polygon)-1


    for i in range(len(polygon)):

        xi,yi=polygon[i]

        xj,yj=polygon[j]


        if (
            ((yi>y)!=(yj>y))
            and
            (
                x <
                (xj-xi)
                *
                (y-yi)
                /
                (yj-yi)
                +
                xi
            )
        ):
            inside = not inside


        j=i


    return inside



# ==========================================================
# Classification
# ==========================================================

def duval_pentagon_zone(
        row,
        pentagon="P2"
):

    h2=row.get(
        "duval_pent_pct_h2",
        np.nan
    )

    ch4=row.get(
        "duval_pent_pct_ch4",
        np.nan
    )

    c2h6=row.get(
        "duval_pent_pct_c2h6",
        np.nan
    )

    c2h4=row.get(
        "duval_pent_pct_c2h4",
        np.nan
    )

    c2h2=row.get(
        "duval_pent_pct_c2h2",
        np.nan
    )


    if any(
        pd.isna(x)
        for x in [
            h2,
            ch4,
            c2h6,
            c2h4,
            c2h2
        ]
    ):
        return "Uncertain"



    xy = duval_pentagon_centroid(
        h2,
        ch4,
        c2h6,
        c2h4,
        c2h2
    )


    if xy is None:
        return "Uncertain"



    x,y=xy



    zones = (
        PENTAGON_1_ZONES
        if pentagon=="P1"
        else PENTAGON_2_ZONES
    )


    for name,polygon in zones.items():

        if point_inside_polygon(
            x,
            y,
            polygon
        ):
            return name


    return "Uncertain"



# ==========================================================
# Apply dataframe
# ==========================================================

def apply_duval_pentagon(
        df,
        pentagon="P2"
):


    coords=df.apply(

        lambda r:
        duval_pentagon_centroid(

            r["duval_pent_pct_h2"],
            r["duval_pent_pct_ch4"],
            r["duval_pent_pct_c2h6"],
            r["duval_pent_pct_c2h4"],
            r["duval_pent_pct_c2h2"]

        ),

        axis=1
    )


    df["duval_pentagon_x"] = [
        c[0] if c else np.nan
        for c in coords
    ]

    df["duval_pentagon_y"] = [
        c[1] if c else np.nan
        for c in coords
    ]


    df["duval_pentagon_fault"] = df.apply(

        lambda r:
        duval_pentagon_zone(
            r,
            pentagon
        ),

        axis=1
    )


    return df