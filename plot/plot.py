""" Created on Tue Aug 8 13:23:29 2025
    @author: dcupolillo """

import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.animation import FuncAnimation
from matplotlib.gridspec import GridSpec
from scipy.stats import sem


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
    if not isinstance(spines, list):
        raise TypeError("spines must be a list of dictionaries.")

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
    binary: np.ndarray,
    vmin: float = 0.0,
    vmax: float = 1.0,
    cmap: str = "viridis",
    color: str = "gray",
    scalebar_length: float = 0.25,
    scalebar_x: float = 13.0,
    scalebar_y_start: float = 0.3,
    colorbar_label: str = r"$\Delta F / F_0$"
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

    # Select trials where binary == 1
    spine_sweep_indices = np.argwhere(binary == 1)
    selected_data = data[spine_sweep_indices[:, 0], spine_sweep_indices[:, 1], :]

    mean_trace = np.mean(selected_data, axis=0)
    sem_trace = sem(selected_data, axis=0, nan_policy='omit')

    # Use first trial's timestamps for x-axis
    timestamps = ts[0][0]
    frame_rate = 16
    tick_interval = frame_rate
    tick_indices = np.arange(0, len(timestamps), tick_interval)
    tick_labels = timestamps[tick_indices].round(1)

    fig = plt.figure(figsize=(5, 9))
    gs = GridSpec(
        2, 2,
        width_ratios=[1, 0.05],
        height_ratios=[0.2, 1],
        hspace=0.05, wspace=0.05,
        top=0.99, bottom=0.08)

    ax0 = fig.add_subplot(gs[0, 0])  # mean trace
    ax1 = fig.add_subplot(gs[1, 0])  # heatmap
    cax = fig.add_subplot(gs[1, 1])  # colorbar

    # Mean trace
    ax0.plot(np.arange(len(mean_trace)), mean_trace, color="black")
    ax0.fill_between(
        np.arange(len(sem_trace)),
        mean_trace - sem_trace,
        mean_trace + sem_trace,
        color=color, alpha=0.4)

    ax0.plot(
        [scalebar_x, scalebar_x],
        [scalebar_y_start, scalebar_y_start + scalebar_length],
        color="black", lw=2.)

    for spine in ax0.spines.values():
        spine.set_visible(False)
    ax0.set(ylim=(0, 1), yticks=[], xticks=[], xlim=(0, len(sem_trace)))
    ax0.tick_params("both", length=0)

    # Heatmap
    im = ax1.imshow(
        selected_data,
        aspect='auto', cmap=cmap,
        vmin=vmin, vmax=vmax)

    # Colorbar
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label(colorbar_label, labelpad=-2)
    cbar.ax.set(
        yticks=np.linspace(vmin, vmax, 4),
        yticklabels=np.linspace(vmin, vmax, 4))

    # Heatmap ticks and labels
    ax1.set(
        xlabel="Time around stim. (s)",
        ylim=(0, len(selected_data) - 1),
        yticks=np.linspace(0, selected_data.shape[0] - 1, 2),
        yticklabels=np.linspace(1, selected_data.shape[0], 2, dtype=int),
        xticks=tick_indices,
        xticklabels=tick_labels)
    ax1.set_ylabel("Trials", labelpad=-2)

    ax_top = ax1.secondary_xaxis('top')
    ax_top.set_xticks([tick_indices[1]] if len(tick_indices) > 1 else [])
    ax_top.set_xticklabels([''] if len(tick_indices) > 1 else [])
    ax_top.tick_params(axis='x', direction='out', top=True, labeltop=True)

    return fig, (ax0, ax1)
