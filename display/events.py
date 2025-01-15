""" Created on Thu Oct 26 15:33:29 2023
    @author: dcupolillo """


import numpy as np
import matplotlib.pyplot as plt
from dataplotter import ElectrophyPlotter
from tqdm import tqdm
from scipy.spatial import Voronoi
from spyne.neuralnetwork.spine_segmentation.spine_segmentation import (
    calculate_centroid)
from spyne.core.utils.utils import transform


def plot_spine_pixel_annotation(
        spines: object,
        base_image: np.ndarray,
        image_cmap: str,
        spines_cmap: str,
        spine_mask_alpha: float,
) -> None:
    """
    Plots the segmented spine contours within the base image.

    Parameters
    ----------
    base_image : np.ndarray
        The base image to plot.
    cmap : str
        Color map of the base image.
    spines : object
        Spine collection object containing spine data.
    spines_cmap : str
        Color map of the different spines.
    alpha : float
        Transparency level for the spine masks.

    Returns
    -------
    None
    """

    fig, ax = plt.subplots()
    print(type(spines))

    ax.imshow(base_image, cmap=image_cmap)
    ax.set_aspect('equal')

    cmap = plt.get_cmap(spines_cmap, spines.n_spines)

    for i, spine in enumerate(spines):

        color = cmap(i)

        # Normalize the mask to make sure it is binary (0 or 1)
        mask_normalized = spine.mask.astype(bool)

        shade_color = (color[0],
                       color[1],
                       color[2],
                       spine_mask_alpha)
        colored_mask = np.zeros((*mask_normalized.shape, 4))
        colored_mask[mask_normalized] = shade_color

        # Overlay this colored mask onto the existing image in the axes
        ax.imshow(colored_mask, interpolation='none')

        ax.scatter(spine.centroid_pix[0],
                   spine.centroid_pix[1],
                   color=color)

        ax.text(spine.centroid_pix[0], spine.centroid_pix[1],
                str(i + 1), color=color, fontsize=12,
                ha='center', va='bottom')


def dummy_spine(spines_metadata, roi_metadata):

    centroids = np.array(
        [spine['centroid_pix'] for spine in spines_metadata])
    masks = [spine['mask'] for spine in spines_metadata]

    average_area = int(np.mean([np.sum(mask > 0) for mask in masks]))

    dummy_mask = np.zeros_like(masks[0])
    dummy_height = int(np.sqrt(average_area))
    dummy_width = dummy_height

    vor = Voronoi(centroids)

    largest_region = None
    largest_area = 0
    regions = [vor.regions[i] for i in vor.point_region]

    for region in regions:
        if -1 not in region and len(region) > 0:  # Exclude infinite regions

            polygon = np.array([vor.vertices[i] for i in region])
            area = 0.5 * np.abs(np.dot(
                polygon[:, 0], np.roll(polygon[:, 1], 1)) -
                np.dot(polygon[:, 1], np.roll(polygon[:, 0], 1)))

            if area > largest_area:
                largest_area = area
                largest_region = polygon

    if largest_region is not None:
        dummy_x, dummy_y = largest_region.mean(axis=0).astype(int)

        dummy_x_start = max(0, dummy_x - dummy_width // 2)
        dummy_y_start = max(0, dummy_y - dummy_height // 2)
        dummy_x_end = min(dummy_mask.shape[1], dummy_x_start + dummy_width)
        dummy_y_end = min(dummy_mask.shape[0], dummy_y_start + dummy_height)

        dummy_mask[
            dummy_y_start:dummy_y_end,
            dummy_x_start:dummy_x_end
        ] = 1

    centroid_pix = calculate_centroid(dummy_mask)
    centroid_fov = transform(
        centroid_pix,
        np.array(roi_metadata['affine']),
        np.array(roi_metadata['pixel_to_ref_transform']),
        roi_metadata['center_xy'],
        roi_metadata['pixel_resolution_xy'])

    dummy_spine_dict = {
        'centroid_fov': centroid_fov,
        'centroid_pix': centroid_pix,
        'mask': dummy_mask,
        'roi_n': spines_metadata[0]['roi_n'],
        'roi_z': spines_metadata[0]['roi_z']}

    return dummy_spine_dict


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

        traces = ElectrophyPlotter(
            figsize=(8, fig_height),
            grid=(1, 2),
            font_size=fontsize,
            x_label='Time (s)',
            y_label='Spines',
            hidden_axis=['top', 'right'],
            hide_xticks=False,
            hide_yticks=False,
            spine_offset=20,
            fontsize=fontsize,
            )

        for i, spine in enumerate(
                tqdm(spines,
                     desc="Plotting spines calcium traces",
                     total=spines.n_spines)):

            color = cmap(i)

            for sweep_BLA, ts_BLA, sweep_CA3, ts_CA3 in zip(
                    dFF_BLA[i], timestamps_BLA[i],
                    dFF_CA3[i], timestamps_CA3[i]):

                traces.plot_data(
                    ts_BLA,
                    sweep_BLA + y_offsets[i],
                    row=0,
                    col=0,
                    color=sweep_color,
                    alpha=sweep_alpha,
                    linewidth=sweep_linewidth,
                    y_lims=(y_offsets[-1], y_offsets[0]),
                    x_lims=(0, 3.5),
                    clip_on=False,
                    )

                traces.plot_data(
                    ts_CA3,
                    sweep_CA3 + y_offsets[i],
                    row=0,
                    col=1,
                    color=sweep_color,
                    alpha=sweep_alpha,
                    linewidth=sweep_linewidth,
                    y_lims=(y_offsets[-1], y_offsets[0]),
                    x_lims=(0, 3.5),
                    clip_on=False,
                    )

            traces.plot_data(
                ts_BLA,
                np.mean(dFF_BLA[i], axis=0) + y_offsets[i],
                row=0,
                col=0,
                color=color,
                y_lims=(y_offsets[-1], y_offsets[0]),
                x_lims=(0, 4),
                clip_on=False,
                linewidth=trace_linewidth,
                )

            traces.plot_data(
                ts_CA3,
                np.mean(dFF_CA3[i], axis=0) + y_offsets[i],
                row=0,
                col=1,
                color=color,
                y_lims=(y_offsets[-1], y_offsets[0]),
                x_lims=(0, 4),
                clip_on=False,
                linewidth=trace_linewidth,
                )

        traces.add_vertical_scale_bar(
            ax=traces.ax,
            y_unit=scalebar_y_unit,
            position=(3.5, spines.n_spines / 2 * increment),
            y_label=fr'{scalebar_y_unit} $\Delta F/F_0$',
            linewidth=2,
            fontsize=fontsize,
            row=0,
            col=1
            )

        for ax in traces.axes:
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

        traces = ElectrophyPlotter(
            figsize=(4, fig_height),
            font_size=fontsize,
            x_label='Time (s)',
            y_label='Spines',
            hidden_axis=['top', 'right'],
            hide_xticks=False,
            hide_yticks=False,
            spine_offset=20,
            fontsize=fontsize,
            )

        for i, spine in enumerate(
                tqdm(spines,
                     desc="Plotting spines calcium traces",
                     total=spines.n_spines)):

            color = cmap(i)

            for sweep, ts in zip(dFF[i], timestamps[i]):
                traces.plot_data(
                    ts,
                    sweep + y_offsets[i],
                    color=sweep_color,
                    alpha=sweep_alpha,
                    linewidth=sweep_linewidth,
                    y_lims=(y_offsets[-1], y_offsets[0]),
                    x_lims=(0, 3.5),
                    clip_on=False,
                    )

            traces.plot_data(
                ts,
                np.mean(dFF[i], axis=0) + y_offsets[i],
                color=color,
                y_lims=(y_offsets[-1], y_offsets[0]),
                x_lims=(0, 4),
                clip_on=False,
                linewidth=trace_linewidth,
                )

        traces.add_vertical_scale_bar(
            ax=traces.ax,
            y_unit=scalebar_y_unit,
            position=(3.5, spines.n_spines / 2 * increment),
            y_label=fr'{scalebar_y_unit} $\Delta F/F_0$',
            linewidth=2,
            fontsize=fontsize,
            )

        traces.ax.axvline(1, color='red', alpha=0.6, lw=2, clip_on=False)

        traces.ax.set_yticks(y_offsets)
        traces.ax.set_yticklabels(list(range(1, spines.n_spines + 1)))


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

    fig_zcore, ax_zscore = plt.subplots()

    for i, spine in enumerate(
            tqdm(spines,
                 desc="Plotting spines z-scores",
                 total=spines.n_spines)):

        # for sweep, ts in zip(zscore[i], timestamps[i]):

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
    ax_zscore.set_xlabel('Time (s)')
    ax_zscore.set_ylabel('Sweeps of spines')
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

        traces = ElectrophyPlotter(
            figsize=(10, 5),
            grid=(2, 2),
            height_ratios=(1, 2),
            font_size=fontsize,
            x_label='Time (s)',
            y_label='Sweeps',
            hidden_axis=['top', 'right'],
            hide_xticks=False,
            hide_yticks=False,
            spine_offset=20,
            )

        y_offsets = np.linspace(
            0, len(dFF_BLA) * increment, len(dFF_BLA))

        for i, (sweep_BLA, sweep_CA3, ts_BLA, ts_CA3) in enumerate(zip(
                dFF_BLA, dFF_CA3, timestamps_BLA, timestamps_CA3)):

            traces.plot_data(
                ts_BLA,
                sweep_BLA,
                row=0,
                col=0,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                y_lims=(y_offsets[0], y_offsets[-1]),
                x_lims=(0, 4),
                clip_on=False,
                )

            traces.plot_data(
                ts_CA3,
                sweep_CA3,
                row=0,
                col=1,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                y_lims=(y_offsets[0], y_offsets[-1]),
                x_lims=(0, 4),
                clip_on=False,
                )

            traces.plot_data(
                ts_BLA,
                sweep_BLA + y_offsets[i],
                row=1,
                col=0,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                y_lims=(y_offsets[0], y_offsets[-1]),
                x_lims=(0, 4),
                clip_on=False,
                )

            traces.plot_data(
                ts_CA3,
                sweep_CA3 + y_offsets[i],
                row=1,
                col=1,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                y_lims=(y_offsets[0], y_offsets[-1]),
                x_lims=(0, 4),
                clip_on=False,
                )

        traces.plot_data(
            ts_BLA,
            np.mean(dFF_BLA, axis=0),
            row=0,
            col=0,
            color=mean_color,
            alpha=mean_alpha,
            linewidth=mean_linewidth,
            y_lims=(np.min(dFF_BLA) - 0.1, np.max(dFF_BLA) + 0.1),
            x_lims=(0, 4),
            clip_on=False,
            )
        traces.axes[0, 0].spines[['bottom', 'left']].set_visible(False)
        traces.axes[0, 0].set_xticks([])
        traces.axes[0, 0].set_xticklabels([])
        traces.axes[0, 0].set_xlabel('')
        traces.axes[0, 0].set_yticks([])
        traces.axes[0, 0].set_yticklabels([])
        traces.axes[0, 0].set_ylabel('')

        traces.plot_data(
            ts_CA3,
            np.mean(dFF_CA3, axis=0),
            row=0,
            col=1,
            color=mean_color,
            alpha=mean_alpha,
            linewidth=mean_linewidth,
            y_lims=(np.min(dFF_CA3) - 0.1, np.max(dFF_CA3) + 0.1),
            x_lims=(0, 4),
            clip_on=False,
            )
        traces.axes[0, 1].spines[['bottom', 'left']].set_visible(False)
        traces.axes[0, 1].set_xticks([])
        traces.axes[0, 1].set_xticklabels([])
        traces.axes[0, 1].set_xlabel('')
        traces.axes[0, 1].set_yticks([])
        traces.axes[0, 1].set_yticklabels([])
        traces.axes[0, 1].set_ylabel('')

        traces.add_vertical_scale_bar(
            ax=traces.axes[0, 1],
            y_unit=scalebar_y_unit,
            position=(3.5, np.mean(dFF_BLA[0])),
            y_label=fr'{scalebar_y_unit} $\Delta F/F_0$',
            linewidth=2,
            fontsize=fontsize,
            row=0,
            col=1
            )

        traces.add_vertical_scale_bar(
            ax=traces.axes[1, 1],
            y_unit=scalebar_y_unit,
            position=(3.5, len(dFF_BLA) / 2 * increment),
            y_label=fr'{scalebar_y_unit} $\Delta F/F_0$',
            linewidth=2,
            fontsize=fontsize,
            row=0,
            col=1
            )

        for ax in traces.axes[0]:
            ax.axvline(
                1, color='red', alpha=0.6, lw=2, clip_on=False)

        for ax in traces.axes[1]:
            ax.axvline(
                1, color='red', alpha=0.6, lw=2, clip_on=False)

            ax.set_yticks(y_offsets)
            ax.set_yticklabels(
                list(range(1, len(dFF_BLA) + 1)))

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

        traces = ElectrophyPlotter(
            figsize=(5, 5),
            grid=(2, 1),
            height_ratios=(1, 2),
            font_size=fontsize,
            x_label='Time (s)',
            y_label='Sweeps',
            hidden_axis=['top', 'right'],
            hide_xticks=False,
            hide_yticks=False,
            spine_offset=20,
            fontsize=fontsize,
            )

        y_offsets = np.linspace(
            0, len(dFF) * increment, len(dFF))

        for i, (sweep, ts) in enumerate(zip(dFF, timestamps)):

            traces.plot_data(
                ts,
                sweep,
                row=0,
                col=0,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                y_lims=(np.min(dFF) - 0.1, np.max(dFF) + 0.1),
                x_lims=(0, 4),
                clip_on=False,
                )

            traces.plot_data(
                ts,
                sweep + y_offsets[i],
                row=1,
                col=0,
                color=sweep_color,
                alpha=sweep_alpha,
                linewidth=sweep_linewidth,
                y_lims=(y_offsets[0], y_offsets[-1]),
                x_lims=(0, 4),
                clip_on=False,
                )

        traces.plot_data(
            ts,
            np.mean(dFF, axis=0),
            row=0,
            col=0,
            color=mean_color,
            y_lims=(np.min(dFF) - 0.1, np.max(dFF) + 0.1),
            x_lims=(0, 4),
            clip_on=False,
            linewidth=mean_linewidth,
            )

        traces.add_vertical_scale_bar(
            ax=traces.axes[0],
            y_unit=scalebar_y_unit,
            position=(3.5, np.mean(dFF[0])),
            y_label=fr'{scalebar_y_unit} $\Delta F/F_0$',
            linewidth=2,
            fontsize=fontsize,
            )

        traces.add_vertical_scale_bar(
            ax=traces.axes[1],
            y_unit=scalebar_y_unit,
            position=(3.5, len(dFF) / 2 * increment),
            y_label=fr'{scalebar_y_unit} $\Delta F/F_0$',
            linewidth=2,
            fontsize=fontsize,
            )

        for ax in traces.axes:
            ax.axvline(
                1, color='red', alpha=0.6, lw=2, clip_on=False)

        traces.axes[1].set_yticks(y_offsets)
        traces.axes[1].set_yticklabels(
            list(range(1, len(dFF) + 1)))

        traces.axes[0].spines[['bottom', 'left']].set_visible(False)
        traces.axes[0].set_xticks([])
        traces.axes[0].set_xticklabels([])
        traces.axes[0].set_xlabel('')
        traces.axes[0].set_yticks([])
        traces.axes[0].set_yticklabels([])
        traces.axes[0].set_ylabel('')
