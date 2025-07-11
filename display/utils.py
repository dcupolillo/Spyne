""" Created on Fri Feb  7 12:53:28 2025
    @author: dcupolillo """

import numpy as np
import matplotlib.pyplot as plt


def optimal_subplot_layout(n: int) -> tuple:
    rows = int(np.ceil(np.sqrt(n)))
    cols = int(np.ceil(n / rows))

    return rows, cols


def L_scalebar(
        ax: plt.Axes,
        hline_length: float,
        vline_length: float,
        hline_left_edge: float,
        vline_bottom_edge: float,
        x_units: str,
        y_units: str,
        hpad: float,
        vpad: float,
        fontsize: int = 12,
        color: str = "black",
        lw: float = 1.5,
        L_direction: str = "left",
) -> None:

    if L_direction not in ["left", "right"]:
        raise ValueError(
            "Unrecognized L_direction. Should be either `left` or `right`.")

    hline_bounds = (hline_left_edge, hline_left_edge + hline_length)
    vline_bounds = (vline_bottom_edge, vline_bottom_edge + vline_length)

    ax.vlines(
        x=hline_bounds[1] if L_direction == "left" else hline_bounds[0],
        ymin=vline_bounds[0],
        ymax=vline_bounds[1],
        color=color,
        lw=lw)

    ax.hlines(
        y=vline_bounds[0],
        xmin=hline_bounds[0],
        xmax=hline_bounds[1],
        color=color,
        lw=lw)

    ax.text(
        x=np.mean(hline_bounds),
        y=vline_bounds[0] + vpad,
        s=f"{hline_length} {x_units}",
        verticalalignment="top",
        horizontalalignment="center",
        fontsize=fontsize)

    ax.text(
        x=(hline_bounds[1] + hpad
           if L_direction == "left"
           else hline_bounds[0] - hpad),
        y=np.mean(vline_bounds),
        s=f"{vline_length} {y_units}",
        verticalalignment="center",
        horizontalalignment=L_direction,
        fontsize=fontsize,
        rotation=90 if L_direction == "left" else 270)
