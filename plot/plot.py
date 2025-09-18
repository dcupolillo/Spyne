""" Created on Tue Aug 8 13:23:29 2025
    @author: dcupolillo """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from matplotlib.transforms import Affine2D
from scipy.stats import sem
from spyne.core.imagingdataset import Roi
from spyne.core.utils.spine_node import euclidean_distance


def _collect_centroid_fov(
        data: list,
        scan_angle: bool,
) -> list:
    """
    Collects all 'centroid_fov' tuples from a list of spine dictionaries.

    Parameters
    ----------
    data : list
        A list of dictionaries containing 'centroid_fov' keys.
    scan_angle : bool
        If True, use scan angle degree units, else micrometers.

    Returns
    -------
    list
        A list of 'centroid_fov' tuples.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of dictionaries.")

    centroid_fov_list = [None] * len(data)
    unit = 'centroid_fov' if scan_angle else 'centroid_fov_um'

    for i, spine in enumerate(data):
        centroid_fov_list[i] = tuple(spine[unit])

    return centroid_fov_list


def _collect_centroid_fov_3d(
        data: list,
        scan_angle: bool,
) -> list:
    """
    Collects all 3D 'centroid_fov' coordinates from spine dictionaries.

    Parameters
    ----------
    data : list
        A list of dictionaries containing 'centroid_fov' keys.
    scan_angle : bool
        If True, use scan angle degree units, else micrometers.

    Returns
    -------
    list
        A list of 3D 'centroid_fov' tuples including z-coordinate.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of dictionaries.")

    centroid_fov_list = [None] * len(data)
    unit = 'centroid_fov' if scan_angle else 'centroid_fov_um'

    for i, spine in enumerate(data):
        centroid_fov_list[i] = tuple(
            spine[unit] + [spine['roi_z']])

    return centroid_fov_list


def add_hscalebar(
        x_start: float,
        x_length: float,
        y_position: float,
        ax: plt.Axes,
        legend: str = None,
        legend_offset: float = 0.1,
        **kwargs
) -> None:

    default_kwargs = {
        "color": "black",
        "linewidth": 2
    }

    scalebar_kwargs = {**default_kwargs, **kwargs}

    # Add horizontal scale bar
    ax.hlines(
        y=y_position,
        xmin=x_start, xmax=x_start + x_length,
        **scalebar_kwargs
    )

    if legend:
        ax.text(
            x_start + x_length / 2,
            y_position + legend_offset,
            legend,
            ha='center',
            va='bottom' if legend_offset > 0 else 'top',
        )


def add_vscalebar(
        y_start: float,
        y_length: float,
        x_position: float,
        ax: plt.Axes,
        legend: str = None,
        legend_offset: float = 0.1,
        **kwargs
) -> None:

    default_kwargs = {
        "color": "black",
        "linewidth": 2
    }

    scalebar_kwargs = {**default_kwargs, **kwargs}

    # Add vertical scale bar
    ax.vlines(
        x=x_position,
        ymin=y_start, ymax=y_start + y_length,
        **scalebar_kwargs
    )

    if legend:
        ax.text(
            y_start + y_length / 2,
            x_position + legend_offset,
            legend,
            ha='left' if legend_offset > 0 else 'right',
            va='center',
        )


def scatter(
        spines: list,
        scan_angle: bool = False,
        projection: str = '2d',
        ax: plt.Axes = None,
        **kwargs,
) -> None:
    """
    Plot detected spines as scatter points in 2D or 3D.

    Parameters
    ----------
    spines : list
        List of spine dictionaries containing centroid coordinates.
    scan_angle : bool, optional
        Whether to use scan angle coordinates. Default is False.
    projection : str, optional
        Plot projection type: '2d' for 2D scatter, '3d' for 3D scatter.
        Default is '2d'.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Default is None.
    **kwargs
        Additional arguments passed directly to plt.scatter.
        Common options include:
        - s : size of markers
        - color or c : marker color
        - edgecolors : edge color of markers
        - alpha : transparency
        - marker : marker style
        - fontsize : font size for labels

    Returns
    -------
    None
    """
    if not isinstance(spines, (list, dict)):
        raise TypeError("spines must be a list of dictionaries.")

    if isinstance(spines, dict):
        spines = [spines]

    if projection not in ['2d', '3d']:
        raise ValueError("projection must be '2d' or '3d'")

    # Get coordinates based on projection type
    if projection == '2d':
        all_centroids = np.array(
            _collect_centroid_fov(spines, scan_angle=scan_angle))
    else:
        all_centroids = np.array(
            _collect_centroid_fov_3d(spines, scan_angle=scan_angle))

    if len(all_centroids) == 0:
        raise ValueError("No spine data found.")

    if ax is None:
        if projection == '3d':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            _, ax = plt.subplots()

        # Set labels based on projection
        ax.set_xlabel('X coordinate')
        ax.set_ylabel('Y coordinate')
        ax.set_aspect('equal')

        if projection == '3d':
            ax.set_zlabel('Z coordinate')

    # Plot based on projection
    if projection == '2d':
        ax.scatter(
            all_centroids[:, 0],  # xs of all centroids
            all_centroids[:, 1],  # ys of all centroids
            **kwargs)
    else:
        ax.scatter(
            all_centroids[:, 0],  # xs of all centroids
            all_centroids[:, 1],  # ys of all centroids
            all_centroids[:, 2],  # zs of all centroids
            **kwargs)


def scatter_events(
        spines: list,
        events: np.ndarray,
        n_event_threshold: int = 0,
        scan_angle: bool = False,
        ax: plt.Axes = None,
        projection: str = '2d',
        cmap: str = 'viridis',
        show_cbar: bool = True,
        cbar_kwargs: dict = None,
        **kwargs
) -> None or tuple:
    """
    Plot active spines with event-based coloring.

    Parameters
    ----------
    spines : list
        List of spine dictionaries containing centroid coordinates.
    events : np.ndarray
        Array of event data corresponding to each spine.
    n_event_threshold : int, optional
        Minimum events for active spines. Default is 0.
    scan_angle : bool, optional
        Whether to use scan angle coordinates. Default is False.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Default is None.
    projection : str, optional
        Projection type for the plot. Default is '2d'.
    cmap : str, optional
        Colormap for event representation. Default is 'viridis'.
    show_cbar : bool, optional
        Whether to show colorbar. Default is True.
    cbar_kwargs : dict, optional
        Additional arguments passed to plt.colorbar.
        Common options include:
        - shrink : colorbar size scaling (default: 0.8)
        - label : colorbar label (default: 'Number of Events')
        - orientation : 'vertical' or 'horizontal'
        - pad : padding between axes and colorbar
        - aspect : aspect ratio of colorbar
    **kwargs
        Additional arguments passed to scatter plot.
        Common options include:
        - s : size of markers (default: 20)
        - edgecolors : edge color of markers
        - alpha : transparency
        - marker : marker style
        - fontsize : font size for labels (default: 14)
        - zorder : plot layer order

    Returns
    -------
    tuple of (scatter, colorbar) if show_cbar is True, else None
        scatter : matplotlib PathCollection
            The scatter plot object for further customization.
        colorbar : matplotlib Colorbar
            The colorbar object for further customization.
    """
    if not isinstance(spines, list):
        raise TypeError("spines must be a list of dictionaries.")

    if not isinstance(events, np.ndarray):
        raise TypeError("events must be a numpy array.")

    if len(spines) != events.shape[0]:
        raise ValueError("spines and events must have the same length.")

    if events.shape[0] > 0 and n_event_threshold > events.shape[1]:
        raise ValueError(
            f"n_event_threshold should be <= {len(events[0])}")

    if projection not in ['2d', '3d']:
        raise ValueError("projection must be '2d' or '3d'")

    event_counts = events.sum(axis=1)

    # Collect centroids based on projection type
    if projection == '2d':
        all_centroids = np.array(
            _collect_centroid_fov(spines, scan_angle=scan_angle))
    else:
        all_centroids = np.array(
            _collect_centroid_fov(spines, scan_angle=scan_angle))

    if len(all_centroids) == 0:
        return

    # Filter out spines with sufficient events
    mask = event_counts > n_event_threshold
    if not np.any(mask):
        return None

    centroids = all_centroids[mask]
    counts = event_counts[mask]

    if len(centroids) == 0:
        return

    if ax is None:
        if projection == '3d':
            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
        else:
            _, ax = plt.subplots(
                layout="constrained")

    ax.set_xlabel('X coordinate')
    ax.set_ylabel('Y coordinate')
    if projection == '3d':
        ax.set_zlabel('Z coordinate')
    ax.set_aspect('equal')
    ax.tick_params("both")

    if projection == '2d':
        scatter = ax.scatter(
            [c[0] for c in centroids],  # X coordinates
            [c[1] for c in centroids],  # Y coordinates
            c=counts,
            cmap=cmap,
            **kwargs
        )
    else:
        scatter = ax.scatter(
            [c[0] for c in centroids],  # X coordinates
            [c[1] for c in centroids],  # Y coordinates
            [c[2] for c in centroids],  # Z coordinates
            c=counts,
            cmap=cmap,
            **kwargs
        )

    if show_cbar:
        cbar_kwargs = {} if cbar_kwargs is None else cbar_kwargs
        cbar = plt.colorbar(
            scatter,
            ax=ax,
            **cbar_kwargs
        )

        return scatter, cbar


def masks(
        roi_segmenter,
        image_cmap: str = 'binary_r',
        spines_cmap: str = 'gist_rainbow',
        spine_mask_alpha: float = 0.5,
        enum: bool = True,
        ax: plt.Axes = None,
        output_filename: str or Path = None,
        imshow_kwargs: dict = None,
        scatter_kwargs: dict = None,
        text_kwargs: dict = None,
) -> plt.Axes:
    """
    Plot spine masks for a specific ROI.

    This function overlays segmented spine masks onto a base image for
    visualization. Each spine is represented with a unique color from
    the specified colormap.

    Parameters
    ----------
    roi_segmenter : RoiSegmenter
        The ROI segmenter instance containing spine masks and base image.
        Must have attributes: spines (iterable with mask and centroid_pix),
        base_image (2D array), and n_spines (int).
    image_cmap : str, optional
        Colormap for base image. Default is 'binary_r'.
    spines_cmap : str, optional
        Colormap for spine masks. Default is 'gist_rainbow'.
    spine_mask_alpha : float, optional
        Transparency of spine masks (0-1). Default is 0.5.
    enum : bool, optional
        Whether to enumerate spines with numbers. Default is True.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Default is None.
    output_filename : str or Path, optional
        Path to save figure. Default is None.
    imshow_kwargs : dict, optional
        Additional arguments passed to ax.imshow for base image.
        Common options include:
        - interpolation : image interpolation method
        - aspect : aspect ratio control
        - alpha : image transparency
    scatter_kwargs : dict, optional
        Additional arguments passed to scatter plot for centroids.
        Common options include:
        - s : marker size (default: 50)
        - edgecolors : marker edge color
        - linewidths : marker edge width
        - zorder : plot layer order
    text_kwargs : dict, optional
        Additional arguments passed to text labels.
        Common options include:
        - fontsize : text size (default: 12)
        - fontweight : text weight
        - bbox : text bounding box styling
        - zorder : text layer order
    **kwargs
        Additional arguments for general styling.
        Common options include:
        - fontsize : default font size for ticks/labels (default: 14)

    Returns
    -------
    plt.Axes
        The matplotlib axes with the plotted spine masks.
    """

    imshow_kwargs = {} if imshow_kwargs is None else imshow_kwargs
    scatter_kwargs = {} if scatter_kwargs is None else scatter_kwargs
    text_kwargs = {} if text_kwargs is None else text_kwargs

    if ax is None:
        fig, ax = plt.subplots(layout="constrained")
    else:
        fig = ax.figure

    ax.set_aspect('equal')

    # Display base image
    ax.imshow(roi_segmenter.base_image, cmap=image_cmap, **imshow_kwargs)

    # Get colormap for spines
    cmap = plt.get_cmap(spines_cmap, roi_segmenter.n_spines)

    # Store axis limits for text positioning
    xlim = ax.get_xlim()
    ylim = ax.get_ylim()

    # Overlay each spine mask
    for i, spine in enumerate(roi_segmenter):
        color = cmap(i)

        # Normalize the mask to ensure binary values (0 or 1)
        mask_normalized = spine.mask.astype(bool)

        # Create RGBA color with transparency
        shade_color = (
            color[0],
            color[1],
            color[2],
            spine_mask_alpha
        )

        # Create colored mask overlay
        colored_mask = np.zeros((*mask_normalized.shape, 4))
        colored_mask[mask_normalized] = shade_color

        # Overlay the colored mask
        ax.imshow(colored_mask, interpolation='none')

        if enum:
            # Plot centroid marker
            ax.scatter(
                spine.centroid_pix[0],
                spine.centroid_pix[1],
                color=color,
                **scatter_kwargs
            )

            # Add spine number annotation
            # Clip text position to stay within plot bounds
            x = np.clip(
                spine.centroid_pix[0],
                xlim[0] + 5,
                xlim[1] - 5
            )
            y = np.clip(
                spine.centroid_pix[1],
                ylim[1] + 5,
                ylim[0] - 5
            )

            ax.text(
                x, y,
                str(i + 1),
                color=color,
                **text_kwargs
            )

    # Save figure if requested
    if output_filename is not None:
        ax.axis('off')
        fig.savefig(
            output_filename,
            bbox_inches='tight',
            pad_inches=0
        )
        print(f'Saved! as {output_filename}')

    return ax


def animate_spines(
        spines: list,
        scan_angle: bool = False,
        interval: int = 100,
        frames: int = 100,
        save_path: str = None,
        neuron_orientation: str = 'horizontal',
        elev_start: float = -30,
        elev_end: float = 60,
        azim_start: float = 0,
        azim_end: float = 360,
        use_cmap: bool = False,
        cmap: str = "viridis",
        cbar_kwargs: dict = None,
        **kwargs
) -> FuncAnimation:
    """
    Animate spines in 3D space with rotating viewpoint.

    Parameters
    ----------
    spines : list
        List of spine dictionaries containing centroid coordinates.
    scan_angle : bool, optional
        Whether to use scan angle coordinates. Default is False.
    interval : int, optional
        Frame interval in milliseconds. Default is 100.
    frames : int, optional
        Number of animation frames. Default is 100.
    save_path : str, optional
        Path to save animation. Default is None.
    neuron_orientation : str, optional
        Neuron orientation ('horizontal' or 'vertical').
        Default is 'horizontal'.
    elev_start : float, optional
        Starting elevation angle for 3D animation. Default is -30.
    elev_end : float, optional
        Ending elevation angle for 3D animation. Default is 60.
    azim_start : float, optional
        Starting azimuth angle for 3D animation. Default is 0.
    azim_end : float, optional
        Ending azimuth angle for 3D animation. Default is 360.
    use_cmap : bool, optional
        Whether to use colormap based on z-coordinate. Default is False.
    cmap : str, optional
        Colormap name for z-coordinate coloring. Default is "viridis".
    cbar_kwargs : dict, optional
        Additional arguments passed to plt.colorbar for 3D plots.
        Common options include:
        - label : colorbar label (default: 'Depth')
        - shrink : colorbar size scaling
        - pad : padding between axes and colorbar
    **kwargs
        Additional arguments passed to scatter plot.
        Common options include:
        - s : size of markers (default: 20)
        - color : marker color (default: "grey")
        - edgecolors : edge color of markers (default: 'k')
        - alpha : transparency
        - marker : marker style

    Returns
    -------
    matplotlib.animation.FuncAnimation
        The animation object for display or saving.
    """
    if not isinstance(spines, list):
        raise TypeError("spines must be a list of dictionaries.")

    if neuron_orientation not in ["horizontal", "vertical"]:
        raise ValueError(
            "neuron_orientation must be 'horizontal' or 'vertical'.")

    all_centroids = np.array(
        _collect_centroid_fov_3d(spines, scan_angle=scan_angle))

    if len(all_centroids) == 0:
        raise ValueError("No spine data found for animation.")

    x, y, z = all_centroids[:, 0], all_centroids[:, 1], all_centroids[:, 2]

    # Setup figure and axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Handle neuron orientation for 3D
    if neuron_orientation == "horizontal":
        plot_x, plot_y, plot_z = x, y, z
    else:
        plot_x, plot_y, plot_z = x, z, y

    # Create scatter plot with or without colormap
    if use_cmap:
        scatter = ax.scatter(
            plot_x, plot_y, plot_z,
            c=z, cmap=cmap,
            **kwargs
        )

        # Add colorbar
        norm = plt.Normalize(z.min(), z.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        plt.colorbar(sm, ax=ax, **cbar_kwargs)

    else:
        scatter = ax.scatter(
            plot_x, plot_y, plot_z,
            **kwargs
        )

    # Animation update function
    def update(frame):

        current_elev = (
            elev_start + (elev_end - elev_start) * (frame / (frames - 1)))
        current_azim = (
            azim_start + (azim_end - azim_start) * (frame / (frames - 1)))
        ax.view_init(elev=current_elev, azim=current_azim)

        return scatter,

    # Create animation
    anim = FuncAnimation(
        fig, update, frames=frames, interval=interval, blit=False)

    # Save animation if path is provided
    if save_path:
        anim.save(save_path, writer="pillow")

    return anim


def heatmap(
    data: np.ndarray,
    ts: np.ndarray,
    binary: np.ndarray = None,
    select_binary: int = 1,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    color: str = "gray",
    scalebar_length: float = 0.25,
    scalebar_x: float = 13.0,
    scalebar_y_start: float = 0.3,
    colorbar_label: str = r"$\Delta F / F_0$",
    framerate: float = 16.0
) -> tuple:
    """
    Plot a heatmap and mean trace for selected trials (dFF or zscore).

    Parameters
    ----------
    data : np.ndarray
        Array of shape (n_spines, n_trials, n_timepoints), e.g., dFF or zscore.
    ts : np.ndarray
        Array of timestamps for each trial.
    binary : np.ndarray
        Binary array (n_spines, n_trials) indicating selected trials.
        If not provided, plots all trials. Default is None.
    select_binary : int, optional
        Binary value indicating which trials to select. Default is 1.
    vmin : float, optional
        Minimum value for colormap. Default is 0.0.
    vmax : float, optional
        Maximum value for colormap. Default is 1.0.
    cmap : str, optional
        Colormap for heatmap. Default is "viridis".
    color : str, optional
        Color for SEM shading. Default is "gray".
    scalebar_length : float, optional
        Length of the scalebar in seconds. Default is 1.0.
    scalebar_x : float, optional
        X position of the scalebar. Default is 13.0.
    scalebar_y_start : float, optional
        Y position of the scalebar. Default is 0.3.
    colorbar_label : str, optional
        Label for the colorbar. Default is r"$\Delta F / F_0$".
    framerate : float, optional
        Frame rate for the heatmap animation. Default is 16.

    Returns
    -------
    fig : matplotlib.figure.Figure
        The matplotlib figure object.
    axes : tuple
        Tuple of (ax0, ax1): mean trace axis and heatmap axis.

    Raises
    ------
    ValueError
        If input_type is not None and not "BLA" or "CA3".

    Notes
    -----
    - Pass either dFF or zscore as `data`.
    - The function is flexible for both types.
    """

    if binary is not None:
        spine_sweep_indices = np.argwhere(binary == select_binary)
        data = data[spine_sweep_indices[:, 0], spine_sweep_indices[:, 1], :]

    mean_trace = np.mean(data, axis=0)
    sem_trace = sem(data, axis=0, nan_policy='omit')

    # Use first trial's timestamps for x-axis
    timestamps = ts[0][0]
    tick_interval = framerate
    tick_indices = np.arange(0, len(timestamps), tick_interval)
    tick_labels = timestamps[tick_indices].round(1)

    fig = plt.figure(figsize=(5, 9))
    gs = GridSpec(
        2, 2,
        width_ratios=[1, 0.05],
        height_ratios=[0.2, 1],
        hspace=0.05, wspace=0.05,
        top=0.99, bottom=0.08)

    mean_trace_ax = fig.add_subplot(gs[0, 0])
    heatmap_ax = fig.add_subplot(gs[1, 0])
    cax = fig.add_subplot(gs[1, 1])

    # Mean trace
    mean_trace_ax.plot(np.arange(len(mean_trace)), mean_trace, color="black")
    mean_trace_ax.fill_between(
        np.arange(len(sem_trace)),
        mean_trace - sem_trace,
        mean_trace + sem_trace,
        color=color, alpha=0.4)

    mean_trace_ax.plot(
        [scalebar_x, scalebar_x],
        [scalebar_y_start, scalebar_y_start + scalebar_length],
        color="black", lw=2.)

    for spine in mean_trace_ax.spines.values():
        spine.set_visible(False)
    mean_trace_ax.set(ylim=(0, 1), yticks=[], xticks=[], xlim=(0, len(sem_trace)))
    mean_trace_ax.tick_params("both", length=0)

    # Heatmap
    im = heatmap_ax.imshow(
        data,
        aspect='auto', cmap=cmap,
        vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, labelpad=-2)
    cbar.ax.set(
        yticks=np.linspace(vmin, vmax, 4),
        yticklabels=np.linspace(vmin, vmax, 4))

    # Heatmap ticks and labels
    heatmap_ax.set(
        xlabel="Time around stim. (s)",
        ylim=(0, len(data) - 1),
        yticks=np.linspace(0, data.shape[0] - 1, 2),
        yticklabels=np.linspace(1, data.shape[0], 2, dtype=int),
        xticks=tick_indices,
        xticklabels=tick_labels)
    heatmap_ax.set_ylabel("Trials", labelpad=-2)

    heatmap_secondary_axis = heatmap_ax.secondary_xaxis('top')
    heatmap_secondary_axis.set_xticks([tick_indices[1]] if len(tick_indices) > 1 else [])
    heatmap_secondary_axis.set_xticklabels([''] if len(tick_indices) > 1 else [])
    heatmap_secondary_axis.tick_params(axis='x', direction='out', top=True, labeltop=True)

    return fig, (mean_trace_ax, heatmap_ax)


def tile(
        roi_list: list,
        ax: plt.Axes = None,
        **kwargs
) -> tuple:
    """
    Tile the given regions of interest (ROIs) on the provided axes.

    Parameters
    ----------
    roi_list : list
        List of ROIs to tile.
    ax : plt.Axes, optional
        Matplotlib axes to plot on.
        If None, a new figure and axes are created.
    **kwargs
        Additional keyword arguments passed to imshow.

    Returns
    -------
    tuple
        The figure and axes objects.

    Notes
    -----
    This function expects a list of spyne.core.imagingdataset.Roi objects.
    """

    if not isinstance(roi_list, list) and isinstance(roi_list, Roi):
        roi_list = [roi_list]

    if not all(isinstance(roi, Roi) for roi in roi_list):
        raise TypeError(
            "All elements in roi_list must be"
            "instances of spyne.core.imagingdataset.Roi")

    if ax is None:
        _, ax = plt.subplots()

    default_kwargs = {
        "cmap": "binary_r",
        "alpha": 0.5,
    }
    # Merge default kwargs with user-supplied kwargs (user overrides default)
    imshow_kwargs = {**default_kwargs, **kwargs}

    for roi in roi_list:
        image = roi[0].ch2.maxproj.frame
        pix_to_ref = np.array(roi.roi_metadata["pixel_to_ref_transform"])
        translate = np.array(roi.roi_metadata["translate"])
        full_transform = np.matmul(pix_to_ref, translate)

        transform = Affine2D()
        transform.set_matrix(full_transform)

        ax.imshow(
            image,
            transform=transform + ax.transData,
            **imshow_kwargs
        )

    ax.set_aspect("equal")
    ax.autoscale()


def dendrogram(
        nodes_list: list,
        spine_list: list,
        ax: plt.Axes = None,
        spine_length: float = 1.0,
        alternate_spine_length: bool = False,
        branch_y: float = 0.0,
        branch_kwargs: dict = None,
        spine_kwargs: dict = None,
        spine_head_kwargs: dict = None,
        spine_positions_on_dendrogram: np.ndarray = None
) -> None:
    """
    Create a dendrogram plot from the given dendrite and spines.

    Parameters
    ----------
    nodes_list : list
        List of nodes to include in the dendrogram.
    spine_list : list
        List of spines to include in the dendrogram.
    ax : plt.Axes, optional
        Matplotlib axes to plot on. Default is None.
    spine_length : float, optional
        Length of the spine lines. Default is 1.0.
    alternate_spine_length : bool, optional
        Whether to alternate spine lengths. Default is False.
    branch_y : float, optional
        Y position of the branch line. Default is 0.0.
    branch_kwargs : dict, optional
        Additional arguments passed to the branch line plot.
    spine_kwargs : dict, optional
        Additional arguments passed to the spine lines plot.
    spine_head_kwargs : dict, optional
        Additional arguments passed to the spine head scatter plot.

    Returns
    -------
    None
    """

    # Make sure nodes belong to the same dendrite
    if not all(
            node.branch_id == nodes_list[0].branch_id for node in nodes_list):
        raise ValueError("All nodes must belong to the same dendrite.")

    # Make sure spines are on the same branch
    if not all(
            spine["branch_id"] == spine_list[0]["branch_id"]
            for spine in spine_list):
        raise ValueError("All spines must belong to the same branch.")

    # # Make sure spines are on the same branch
    # if not spine_list[0]["branch_id"] == nodes_list[0].branch_id:
    #     raise ValueError("All spines must belong to the same branch.")

    # Check branch and spine kwargs
    if branch_kwargs is not None and not isinstance(branch_kwargs, dict):
        raise TypeError("branch_kwargs must be a dictionary.")
    if spine_kwargs is not None and not isinstance(spine_kwargs, dict):
        raise TypeError("spine_kwargs must be a dictionary.")

    # Branch kwargs and spine kwargs defaults
    branch_defaults = {
        "color": "black",
        "linewidth": 2,
        "linestyle": "-",
    }
    spine_defaults = {
        "color": "crimson",
        "linewidth": 1.5,
    }
    spine_head_defaults = {
        "color": "crimson",
        "s": 20,
        "edgecolor": "black",
        "linewidth": 0.5,
    }

    # Merge user-supplied kwargs (user overrides default)
    branch_kwargs = {**branch_defaults, **(branch_kwargs or {})}
    spine_kwargs = {**spine_defaults, **(spine_kwargs or {})}
    spine_head_kwargs = {**spine_head_defaults, **(spine_head_kwargs or {})}

    # Extract xyz positions from nodes
    positions = [(node.x, node.y, node.z) for node in nodes_list]
    # Calculate cumulative distances along the branch
    distances = [0]
    for i in range(1, len(positions)):
        d = euclidean_distance(positions[i-1], positions[i])
        distances.append(distances[-1] + d)

    # Map node.id to dendrogram coordinate (distance)
    node_id_to_dendro_coord = {
        node.id: dist for node, dist in zip(nodes_list, distances)}

    # Plot horizontal segment (branch)
    if ax is None:
        _, ax = plt.subplots(figsize=(8, 2), layout="constrained")

    ax.plot(
        distances,
        [branch_y]*len(distances),
        zorder=2,
        clip_on=False,
        **branch_kwargs)

    # Plot spines at their mapped dendrogram coordinate
    if spine_positions_on_dendrogram is not None:

        spine_x = np.asarray(spine_positions_on_dendrogram, dtype=float)
        if spine_x.shape[0] != len(spine_list):
            raise ValueError(
                "spine_positions_on_dendrogram must match len(spine_list).")
        # Optionally clamp to branch extent:
        # spine_x = np.clip(spine_x, distances.min(), distances.max())
    else:
        # default: use closest_node_id mapping
        spine_x = []
        for sp in spine_list:
            cid = sp["closest_node_id"]
            if cid not in node_id_to_dendro_coord:
                raise KeyError(f"closest_node_id {cid} not in nodes_list.")
            spine_x.append(node_id_to_dendro_coord[cid])
        spine_x = np.asarray(spine_x, dtype=float)

    for spine_n, (sp, dendro_x) in enumerate(zip(spine_list, spine_x)):

        this_spine_length = (
            spine_length if not alternate_spine_length
            else (spine_length if spine_n % 2 == 0 else -spine_length))

        y_top = branch_y + this_spine_length

        ax.plot(
            [dendro_x, dendro_x],
            [branch_y, y_top],
            zorder=1,
            clip_on=False,
            **spine_kwargs)
        ax.scatter(
            dendro_x,
            y_top,
            zorder=3,
            clip_on=False,
            **spine_head_kwargs)

    ax.set(
        xlabel='Distance from origin (μm)',
        yticks=[]
    )
    ax.spines[['top', 'right', 'left']].set_visible(False)
    ax.tick_params(axis='y', length=0)
