""" Created on Thu Oct 26 15:33:29 2023
    @author: dcupolillo """


import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from spyne.neuralnetwork.spine_segmentation.spine_segmentation import (
    calculate_centroid)


def _format_axes(
        axes: plt.Axes or np.ndarray,
        fontsize: int = 14,
        hide_axes: list = ["top", "right"],
        offset_axes: list = ["bottom", "left"],
        offset: int = 20,
        set_aspect: bool = True
) -> plt.Axes or np.ndarray:
    """
    Format matplotlib axes with customized appearance settings.

    This function hides specific axes spines, offsets others, and adjusts tick
    label font sizes. It ensures consistent styling across multiple axes.

    Parameters
    ----------
    axes : plt.Axes or np.ndarray
        A single matplotlib Axes object or an array of Axes objects to format.
    fontsize : int, optional
        Font size for tick labels. Default is 14.
    hide_axes : list, optional
        List of axes spines to hide. Default is ["top", "right"].
    offset_axes : list, optional
        List of axes spines to offset outward. Default is ["bottom", "left"].
    offset : int, optional
        Distance (in points) to offset the spines. Default is 20.
    set_aspect : bool, optional
        If True, set the aspect ratio of the axes to 'equal'. Default is True.

    Returns
    -------
    plt.Axes or np.ndarray
        Modified axes object(s).

    Raises
    ------
    ValueError
        If an invalid spine name is included in `hide_axes` or `offset_axes`,
        or if `axes` is not a valid matplotlib Axes object or an array of Axes.
    """

    valid_spines = {"top", "right", "bottom", "left"}

    if not set(hide_axes).issubset(valid_spines):
        raise ValueError(
            f"Invalid spines in hide_axes: {set(hide_axes) - valid_spines}")
    if not set(offset_axes).issubset(valid_spines):
        raise ValueError(
            "Invalid spines in offset_axes: "
            f"{set(offset_axes) - valid_spines}")
    if not isinstance(axes, (plt.Axes, np.ndarray)):
        raise ValueError(
            "Input axes must be a matplotlib Axes object or an array of Axes.")

    if isinstance(axes, plt.Axes):
        axes = [axes]

    axes = np.atleast_1d(axes).flatten()

    for ax in axes:
        for spine in hide_axes:
            ax.spines[spine].set_visible(False)
        for spine in offset_axes:
            ax.spines[spine].set_position(("outward", offset))
        ax.tick_params("both", labelsize=fontsize)

        if set_aspect:
            ax.set_aspect('equal')

    return axes


def plot_spine_pixel_annotation(
        spines: object,
        base_image: np.ndarray,
        image_cmap: str,
        spines_cmap: str,
        spine_mask_alpha: float,
        enum: bool,
        enumerate_fontsize: int,
        fontsize: int,
        ax: plt.Axes,
        output_filename: str or Path,
) -> None:
    """
    Overlay segmented spine masks onto a base image for visualization.

    This function takes a base image and overlays the masks of segmented spines
    onto it. Each spine is represented with a unique color from the specified
    colormap. Spine contours are semi-transparent, allowing the base image to
    remain visible underneath.

    Parameters
    ----------
    spines : object
        Spine collection object containing data for individual spines.
        Each spine should have attributes `mask` (binary mask) and
        `centroid_pix` (coordinates of the centroid).
    base_image : np.ndarray
        The base image over which the spine masks will be overlaid.
    image_cmap : str
        Colormap to apply to the base image.
    spines_cmap : str
        Colormap for coloring the individual spine masks.
    spine_mask_alpha : float
        Transparency level for the spine masks.
        Should be a value between 0 and 1,
        where 0 is fully transparent and 1 is fully opaque.
    enum : bool
        If True, spines will be enumerated on the plot with their indices.
    enumerate_fontsize : int
        Font size for the spine labels.
    fontsize : int
        Font size for axes labels and ticks

    Returns
    -------
    None
        Displays the plot with overlaid spine masks.

    Notes
    -----
    - The function normalizes each spine mask to ensure binary values (0 or 1)
        before applying color.
    - The centroid of each spine is marked and optionally
        annotated with its index.
    """

    if not ax:
        fig, ax = plt.subplots(layout="constrained")
    ax.set_aspect('equal')

    ax.imshow(base_image, cmap=image_cmap)
    ax.tick_params("both", labelsize=fontsize)
    ax.set_xlabel("", fontsize=fontsize)
    ax.set_ylabel("", fontsize=fontsize)

    cmap = plt.get_cmap(spines_cmap, spines.n_spines)

    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    for i, spine in enumerate(spines):

        color = cmap(i)

        # Normalize the mask to make sure it is binary (0 or 1)
        mask_normalized = spine.mask.astype(bool)

        shade_color = (
            color[0],
            color[1],
            color[2],
            spine_mask_alpha)

        colored_mask = np.zeros((*mask_normalized.shape, 4))
        colored_mask[mask_normalized] = shade_color

        # Overlay this colored mask onto the existing image in the axes
        ax.imshow(colored_mask, interpolation='none')

        if enum:
            ax.scatter(
                spine.centroid_pix[0],
                spine.centroid_pix[1],
                color=color)

            x = np.clip(
                spine.centroid_pix[0], xlim[0] + 5, xlim[1] - 5)
            y = np.clip(
                spine.centroid_pix[1], ylim[1] + 5, ylim[0] - 5)

            ax.text(
                x,
                y,
                str(i + 1),
                color=color,
                fontsize=enumerate_fontsize,
                ha='center',
                va='bottom')

    if output_filename is not None:
        ax.axis('off')
        fig.savefig(
            output_filename, bbox_inches='tight', pad_inches=0)
        print(f'Saved! as {output_filename}')


def plot_spine_calcium_traces(
        spines: object,
        input_type: str,
        fontsize: int,
        increment: float,
        spines_cmap: str,
        sweep_color: str,
        sweep_alpha: float,
        sweep_linewidth: float,
        trace_linewidth: float,
        scalebar_y_unit: float,
) -> None:
    """
    Plots calcium traces for each spine.

    Parameters
    ----------
    fontsize : int
        Font size for plot labels and texts.
    spines : object
        Spine collection object containing spine data.
    increment : float
        Increment for y-offsets in the traces plot.
    spines_cmap : str
        Color map of the different spines.
    sweep_color : str
        Color of the sweeps in the traces plot.

    Returns
    -------
    None
    """

    cmap = plt.get_cmap(spines_cmap, spines.n_spines)
    fig_height = spines.n_spines

    y_offsets = np.linspace(
        spines.n_spines * increment, 0, spines.n_spines)

    if not input_type:
        dFF_BLA = spines.dFF_BLA
        timestamps_BLA = spines.ts_BLA
        dFF_CA3 = spines.dFF_CA3
        timestamps_CA3 = spines.ts_BLA

        fig, axes = plt.subplots(
            1, 2,
            figsize=(8, fig_height),
            layout="constrained",
            sharex=True,
            sharey=True)

        _format_axes(axes)

        axes[0].set_xlabel("Time (s)", fontsize=fontsize)
        axes[0].set_ylabel("Spines", fontsize=fontsize)
        axes[0].set_ylim(y_offsets[-1], y_offsets[0])
        axes[0].set_xlim(0, 3.5)

        for i, spine in enumerate(
                tqdm(spines,
                     desc="Plotting spines calcium traces",
                     total=spines.n_spines)):

            color = cmap(i)

            for sweep_BLA, ts_BLA, sweep_CA3, ts_CA3 in zip(
                    dFF_BLA[i], timestamps_BLA[i],
                    dFF_CA3[i], timestamps_CA3[i]):

                axes[0].plot(
                    ts_BLA,
                    sweep_BLA + y_offsets[i],
                    color=sweep_color,
                    alpha=sweep_alpha,
                    linewidth=sweep_linewidth,
                    clip_on=False,
                )
                axes[1].plot(
                    ts_CA3,
                    sweep_CA3 + y_offsets[i],
                    color=sweep_color,
                    alpha=sweep_alpha,
                    linewidth=sweep_linewidth,
                    clip_on=False,
                )

            axes[0].plot(
                ts_BLA,
                np.mean(dFF_BLA[i], axis=0) + y_offsets[i],
                color=color,
                clip_on=False,
                linewidth=trace_linewidth,
            )

            axes[1].plot(
                ts_CA3,
                np.mean(dFF_CA3[i], axis=0) + y_offsets[i],
                color=color,
                clip_on=False,
                linewidth=trace_linewidth,
            )

        ymin = spines.n_spines / 2 * increment,
        ymax = (spines.n_spines / 2 * increment) + scalebar_y_unit,

        for ax in axes:
            ax.vlines(
                x=3.5,
                ymin=ymin,
                ymax=ymax,
                linewidth=2,
            )

            ax.text(
                x=3.5,
                y=np.mean([ymin, ymax]),
                s=fr'{scalebar_y_unit} $\Delta F/F_0$',
            )

            ax.axvline(
                1, color='red', alpha=0.6, lw=2, clip_on=False)

            ax.set_yticks(y_offsets)
            ax.set_yticklabels(
                list(range(1, spines.n_spines + 1)))

    else:

        if input_type == 'CA3':
            dFF = spines.dFF_CA3
            timestamps = spines.ts_CA3
        elif input_type == 'BLA':
            dFF = spines.dFF_BLA
            timestamps = spines.ts_BLA
        else:
            raise ValueError(
                "Invalid input_type. Choose either 'CA3' or 'BLA'.")

        _, ax = plt.subplots(
            figsize=(4, fig_height),
            layout="constrained",
        )

        _format_axes(ax, fontsize=fontsize)
        ax.set_xlabel("Time (s)", fontsize=fontsize)
        ax.set_ylabel("Spines", fontsize=fontsize)
        ax.set_ylims(y_offsets[-1], y_offsets[0]),
        ax.set_xlims(0, 3.5)

        for i, spine in enumerate(
                tqdm(spines,
                     desc="Plotting spines calcium traces",
                     total=spines.n_spines)):

            color = cmap(i)

            for sweep, ts in zip(dFF[i], timestamps[i]):
                ax.plot(
                    ts,
                    sweep + y_offsets[i],
                    color=sweep_color,
                    alpha=sweep_alpha,
                    linewidth=sweep_linewidth,

                    clip_on=False,
                )

            ax.plot(
                ts,
                np.mean(dFF[i], axis=0) + y_offsets[i],
                color=color,
                clip_on=False,
                linewidth=trace_linewidth,
            )

        ymin = spines.n_spines / 2 * increment,
        ymax = (spines.n_spines / 2 * increment) + scalebar_y_unit,

        ax.text(
            x=3.5,
            y=np.mean([ymin, ymax]),
            s=fr'{scalebar_y_unit} $\Delta F/F_0$',
        )

        ax.axvline(1, color='red', alpha=0.6, lw=2, clip_on=False)

        ax.set_yticks(y_offsets)
        ax.set_yticklabels(list(range(1, spines.n_spines + 1)))


def plot_spine_zscores(
        spines: object,
        input_type: str,
        zscore_cmap: str,
        fontsize: int,
) -> None:
    """
    Plots the z-scores for each spine.

    Parameters
    ----------
    spines : object
        Spine collection object containing spine data.

    Returns
    -------
    None
    """

    if input_type == 'CA3':
        zscores = spines.zscore_CA3
        timestamps = spines.ts_CA3
    elif input_type == 'BLA':
        zscores = spines.zscore_BLA
        timestamps = spines.ts_BLA
    else:
        raise ValueError("Invalid input_type. Choose either 'CA3' or 'BLA'.")

    _, ax_zscore = plt.subplots()

    zscores_stacked = np.vstack(zscores)

    im = ax_zscore.imshow(
        zscores_stacked,
        aspect='auto',
        cmap=zscore_cmap,
        extent=[np.min([ts[0] for ts in timestamps]),
                np.max([ts[-1] for ts in timestamps]),
                0,
                spines.n_spines],
        interpolation='none',
    )
    plt.colorbar(im, ax=ax_zscore, label='Z-score')

    ax_zscore.set_xlabel('Time (s)', fontsize=fontsize)
    ax_zscore.set_ylabel('Sweeps of spines', fontsize=fontsize)
    ax_zscore.tick_params("both", labelsize=fontsize)
    ax_zscore.invert_yaxis()


def plot_single_spine_dFF(
        spine: object,
        input_type: str,
        fontsize: int,
        increment: float,
        sweep_color: str,
        sweep_alpha: float,
        mean_color: str,
        mean_alpha: float,
        sweep_linewidth: float,
        mean_linewidth: float,
        scalebar_y_unit: float,
) -> None:

    if not input_type:
        dFF_BLA = spine.dFF_BLA
        timestamps_BLA = spine.ts_BLA
        dFF_CA3 = spine.dFF_CA3
        timestamps_CA3 = spine.ts_BLA

        _, axes = plt.subplots(
            2, 2,
            figsize=(10, 5),
            layout="constrained",
            sharex=True,
            sharey=True,
        )

        _format_axes(ax, fontsize=fontsize)

        axes[0, 0].set_ylims(y_offsets[0], y_offsets[-1])
        axes[0, 0].set_xlims(0, 4)
        axes[0, 0].set_xlabel('Time (s)', fontsize=fontsize)
        axes[0, 0].set_ylabel('Sweeps', fontsize=fontsize)

        y_offsets = np.linspace(
            0, len(dFF_BLA) * increment, len(dFF_BLA))

        for i, (sweep_BLA, sweep_CA3, ts_BLA, ts_CA3) in enumerate(zip(
                dFF_BLA, dFF_CA3, timestamps_BLA, timestamps_CA3)):

            axes[0, 0].plot(
                ts_BLA,
                sweep_BLA,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                clip_on=False,
            )

            axes[0, 1].plot(
                ts_CA3,
                sweep_CA3,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                clip_on=False,
            )

            axes[1, 0].plot(
                ts_BLA,
                sweep_BLA + y_offsets[i],
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                clip_on=False,
            )

            axes[1, 1].plot(
                ts_CA3,
                sweep_CA3 + y_offsets[i],
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                clip_on=False,
            )

        axes[0, 0].plot(
            ts_BLA,
            np.mean(dFF_BLA, axis=0),
            color=mean_color,
            alpha=mean_alpha,
            linewidth=mean_linewidth,
            clip_on=False,
        )

        axes[0, 1](
            ts_CA3,
            np.mean(dFF_CA3, axis=0),
            color=mean_color,
            alpha=mean_alpha,
            linewidth=mean_linewidth,
            clip_on=False,
        )

        for n in [0, 1]:

            ymin = np.mean(dFF[0]) if n == 0 else len(dFF) / 2 * increment
            ymax = ymin + scalebar_y_unit

            axes[0, n].set(
                xticks=[],
                xticklabels=[],
                xlabel='',
                yticks=y_offsets,
                yticklabels=(list(range(1, len(dFF_BLA) + 1))),
                ylabel='')

            axes[n, 1].vlines(
                x=3.5,
                ymin=ymin,
                ymax=ymax,
                linewidth=2,
            )

            ax.text(
                x=3.5,
                y=np.mean([ymin, ymax]),
                s=fr'{scalebar_y_unit} $\Delta F/F_0$',
            )

            axes[n, 0].axvline(
                1, color='red', alpha=0.6, lw=2, clip_on=False)

    else:
        if input_type == 'CA3':
            dFF = spine.dFF_CA3
            timestamps = spine.ts_CA3
        elif input_type == 'BLA':
            dFF = spine.dFF_BLA
            timestamps = spine.ts_BLA
        else:
            raise ValueError(
                "Invalid input_type. Choose either 'CA3' or 'BLA'.")

        _, axes = plt.subplots(
            2, 1,
            figsize=(5, 5),
            layout="constrained",
            sharex=True,
        )

        _format_axes(axes, fontsize=fontsize)

        axes[0].set_ylims(np.min(dFF) - 0.1, np.max(dFF) + 0.1)
        axes[0].set_xlims(0, 4)
        axes[0].set_xlabel('Time (s)', fontsize=fontsize)
        axes[0].set_ylabel('Sweeps', fontsize=fontsize)
        axes[1].set_ylims(y_offsets[0], y_offsets[-1])

        y_offsets = np.linspace(
            0, len(dFF) * increment, len(dFF))

        for i, (sweep, ts) in enumerate(zip(dFF, timestamps)):

            axes[0].plot(
                ts,
                sweep,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                clip_on=False,
            )

            axes[1].plot(
                ts,
                sweep + y_offsets[i],
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                clip_on=False,
            )

        axes[0].plot(
            ts,
            np.mean(dFF, axis=0),
            color=mean_color,
            clip_on=False,
            linewidth=mean_linewidth,
        )

        for n, ax in enumerate(axes):

            ymin = np.mean(dFF[0]) if n == 0 else len(dFF) / 2 * increment
            ymax = ymin + scalebar_y_unit

            ax.vlines(
                x=3.5,
                ymin=ymin,
                ymax=ymax,
                linewidth=2,
            )

            ax.text(
                x=3.5,
                y=np.mean([ymin, ymax]),
                s=fr'{scalebar_y_unit} $\Delta F/F_0$',
                fontsize=fontsize
            )

            ax.axvline(
                1, color='red', alpha=0.6, lw=2, clip_on=False)

        axes[1].set_yticks(y_offsets)
        axes[1].set_yticklabels(
            list(range(1, len(dFF) + 1)))
