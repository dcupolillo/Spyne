""" Created on Mon Jun  3 15:02:25 2024
    @author: dcupolillo """

# This was refractored to the plot/ directory
# to be used as a standalone plotting module
# import spyne.plot as plt_spyne
# this allows for cleaner plotting implementations
# without direct implementation of plotting methods in the segmenter classes

import numpy as np
import math
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy.signal import correlate


def collect_centroid_fov(
        data: list,
        scan_angle: bool,
) -> list:
    """
    Collects all 'centroid_fov' tuples from a 2D list of dictionaries
    into a 1D list.

    Parameters
    ----------
    data : List[List[dict]]
        A 2D list of dictionaries containing 'centroid_fov' keys.
    scan_angle : bool
        If True, the 'centroid_fov' tuples are in scan angle degree units.

    Returns
    -------
    List[Tuple[float, float]]
        A 1D list of 'centroid_fov' tuples.
    """
    if not isinstance(data, list):
        raise TypeError("data must be a list of dictionaries.")

    centroid_fov_list = []

    unit = 'centroid_fov' if scan_angle else 'centroid_fov_um'

    for spine in data:
        centroid_fov_list.append(tuple(
            spine[unit] + [spine['roi_z']]))

    return centroid_fov_list


def collect_centroid_fov_3d(
    spines: list,
    scan_angle: bool
) -> np.ndarray:
    """
    Collects 3D spine centroids (x, y, z) based on the coordinate unit.

    Parameters
    ----------
    spines : list
        List of spine dictionaries with 'centroid_fov' or 'centroid_fov_um' and 'roi_z'.
    scan_angle : bool
        If True, use 'centroid_fov', otherwise use 'centroid_fov_um'.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with spine coordinates.
    """
    unit = 'centroid_fov' if scan_angle else 'centroid_fov_um'
    return np.array([
        spine[unit] + [spine['roi_z']]
        for spine in spines
    ])


def rotate_and_transform_spines(
        spines: list,
        translation: tuple,
        angle: float,
        objective_resolution: float,
) -> list:
    """
    Aligns spines along a specified direction by translating and rotating
    their centroids based on the branch orientation.

    Parameters
    ----------
    branch : list
        List of nodes representing the branch structure.
    spines : list
        List of spine dictionaries containing 'centroid_fov'.
    direction : str
        Direction to align ('horizontal' or 'vertical').
    objective_resolution : float
        Conversion factor from scan angle degrees to micrometers.

    Returns
    -------
    list
        List of transformed spines with updated 'centroid_fov' and
        'centroid_fov_um' coordinates.
    """

    translation_x, translation_y = translation

    # Compute rotation matrix
    cos_theta = math.cos(-angle)
    sin_theta = math.sin(-angle)
    rotation_matrix = np.array(
        [[cos_theta, -sin_theta], [sin_theta, cos_theta]])

    # Create new dictionaries for each spine
    # preventing unwanted in-place mutations
    spines_copy = [{**spine} for spine in spines]

    # Transform each spine
    for spine in spines_copy:
        centroid_x, centroid_y = spine['centroid_fov_um'][:2]

        # Apply translation first
        translated_coords = np.array(
            [centroid_x + translation_x, centroid_y + translation_y])

        # Apply rotation
        rotated_coords = rotation_matrix.dot(translated_coords)

        # Update spine dictionary
        spine['centroid_fov_um'] = [
            round(rotated_coords[0], 6),
            round(rotated_coords[1], 6)]
        spine['centroid_fov'] = [
            spine['centroid_fov_um'][0] / objective_resolution,
            spine['centroid_fov_um'][1] / objective_resolution
        ]

    return spines_copy


def plot_spines_2d(
        spines: list,
        spine_size: int,
        fontsize: int,
        spine_color: str,
        spine_edgecolor: str,
        ax: plt.Axes,
        scan_angle: bool,
        **kwargs: dict,
) -> None:
    """
    Plot all spine centroids from the segmenter as scattered dots.

    Parameters
    ----------
    all_spines : cp.ndarray
        The spine data as a CuPy array.
    spine_size : int, optional
        Size of the scatter points, by default 5.
    spine_color : str or tuple, optional
        Color of the scatter points, by default 'cyan'.
    ax : matplotlib.axes.Axes, optional
        Matplotlib Axes object to plot on. If None, a new figure is created.
    scan_angle : bool, optional
        If True, plot in scan angle degree units. Default is False.

    Raises
    ------
    ValueError
        If no spine data is found in the segmenter.

    Returns
    -------
    None
    """
    if not isinstance(spines, list):
        raise TypeError("spines must be a list of dictionaries.")

    all_centroids = np.array(
        collect_centroid_fov(spines, scan_angle=scan_angle))

    if len(all_centroids) == 0:
        raise ValueError("No spine data found in the segmenter.")

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8, 8),
            constrained_layout=True)

        # Set labels and colors
        ax.set_xlabel('X coordinate', fontsize=fontsize)
        ax.set_ylabel('Y coordinate', fontsize=fontsize)
        ax.tick_params("both", labelsize=fontsize)

        # Set equal aspect ratio
        ax.set_aspect('equal')

    ax.scatter(
        all_centroids[:, 0],  # xs of all centroids in the field-of-view
        all_centroids[:, 1],  # ys of all centroids in the field-of-view
        s=spine_size,
        color=spine_color,
        edgecolors=spine_edgecolor,
        **kwargs)

    plt.show()


def animate_spines_3d(
    figsize: tuple,
    spines: list,
    spine_size: int,
    spine_color: str,
    show_ticks: bool,
    use_cmap: bool,
    cmap: str,
    elev_start: float,
    elev_end: float,
    azim_start: float,
    azim_end: float,
    interval: int,
    frames: int,
    axis_lims: list,
    scan_angle: bool,
    save_path: str,
    neuron_orientation: str,
):
    """
    Animates a 3D scatter plot of spine centroids.

    Parameters
    ----------
    spines : list
        List of spine dictionaries.
    scan_angle : bool
        If True, use 'centroid_fov', otherwise use 'centroid_fov_um'.
    spine_size : int
        Size of the scatter points.
    spine_color : str
        Color of the scatter points.
    use_cmap : bool
        If True, use colormap based on the z-coordinate.
    cmap : str
        Colormap for the z-coordinate.
    elev_start : float
        Starting elevation angle for animation.
    elev_end : float
        Ending elevation angle for animation.
    azim_start : float
        Starting azimuth angle for animation.
    azim_end : float
        Ending azimuth angle for animation.
    interval : int
        Time between frames in milliseconds.
    frames : int
        Number of frames in the animation.
    axis_lims : list
        Axis limits as [xmin, xmax, ymin, ymax, zmin, zmax].
    save_path : str
        Path to save the animation (e.g., .gif or .mp4).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The generated animation object.
    """

    if neuron_orientation not in ["horizontal", "vertical"]:
        raise ValueError("Invalid neuron_orientation. Must be 'horizontal' or 'vertical'.")

    # Collect spine coordinates
    coords = collect_centroid_fov_3d(spines, scan_angle=scan_angle)
    if coords.size == 0:
        raise ValueError("No spine data found for animation.")

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]

    # Setup figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    if use_cmap:
        scatter = ax.scatter(
            x,
            y if neuron_orientation == "horizontal" else z,
            z if neuron_orientation == "horizontal" else y,
            c=z,
            cmap=cmap,
            s=spine_size,
            edgecolors='k')

    else:
        scatter = ax.scatter(
            x,
            y if neuron_orientation == "horizontal" else z,
            z if neuron_orientation == "horizontal" else y,
            color=spine_color,
            s=spine_size,
            edgecolors='k')

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])

    # Set axis limits for equal aspect ratio
    if axis_lims:
        ax.set(
            xlim=(axis_lims[0], axis_lims[1]),
            ylim=(axis_lims[2], axis_lims[3]),
            zlim=(axis_lims[4], axis_lims[5])
        )
    else:
        max_range = np.array([
            max(x) - min(x),
            max(y) - min(y),
            max(z) - min(z)
        ]).max() / 2.0

        mid_x = (max(x) + min(x)) * 0.5
        mid_y = (max(y) + min(y)) * 0.5
        mid_z = (max(z) + min(z)) * 0.5

        if neuron_orientation == "horizontal":
            ax.set(
                xlim=(mid_x - max_range, mid_x + max_range),
                ylim=(mid_y - max_range, mid_y + max_range),
                zlim=(mid_z - max_range, mid_z + max_range)
            )
        else:
            ax.set(
                xlim=(mid_x - max_range, mid_x + max_range),
                zlim=(mid_y - max_range, mid_y + max_range),
                ylim=(mid_z - max_range, mid_z + max_range)
            )

    # Add colorbar
    if use_cmap:
        norm = plt.Normalize(z.min(), z.max())
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        cbar = plt.colorbar(sm, ax=ax)
        cbar.set_label("Depth")

    # Update function for animation
    def update(frame):
        current_elev = elev_start + (elev_end - elev_start) * (frame / (frames - 1))
        current_azim = azim_start + (azim_end - azim_start) * (frame / (frames - 1))
        ax.view_init(elev=current_elev, azim=current_azim)
        return scatter,

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

    # Save animation if path is provided
    if save_path:
        anim.save(save_path, writer="pillow")

    return anim


def animate_event_spines_3d(
    spines: list,
    spines_BLA: list,
    spines_CA3: list,
    figsize: tuple,
    scan_angle: bool,
    spine_size: int,
    spine_color: str,
    spine_edgecolor: str,
    BLA_spine_size: int,
    CA3_spine_size: int,
    BLA_spine_color: str,
    CA3_spine_color: str,
    BLA_spine_edgecolor: str,
    CA3_spine_edgecolor: str,
    show_ticks: bool,
    elev_start: float,
    elev_end: float,
    azim_start: float,
    azim_end: float,
    interval: int,
    frames: int,
    axis_lims: list,
    save_path: str,
    neuron_orientation: str,
):
    """
    Animates a 3D scatter plot of spine centroids.

    Parameters
    ----------
    spines : list
        List of spine dictionaries.
    scan_angle : bool
        If True, use 'centroid_fov', otherwise use 'centroid_fov_um'.
    spine_size : int
        Size of the scatter points.
    spine_color : str
        Color of the scatter points.
    use_cmap : bool
        If True, use colormap based on the z-coordinate.
    cmap : str
        Colormap for the z-coordinate.
    elev_start : float
        Starting elevation angle for animation.
    elev_end : float
        Ending elevation angle for animation.
    azim_start : float
        Starting azimuth angle for animation.
    azim_end : float
        Ending azimuth angle for animation.
    interval : int
        Time between frames in milliseconds.
    frames : int
        Number of frames in the animation.
    axis_lims : list
        Axis limits as [xmin, xmax, ymin, ymax, zmin, zmax].
    save_path : str
        Path to save the animation (e.g., .gif or .mp4).

    Returns
    -------
    anim : matplotlib.animation.FuncAnimation
        The generated animation object.
    """
    if neuron_orientation not in ["horizontal", "vertical"]:
        raise ValueError("Invalid neuron_orientation. Must be 'horizontal' or 'vertical'.")
    
    regular_spines = [
        spine for spine in spines
        if spine not in spines_BLA and
        spine not in spines_CA3]

    # Collect spine coordinates
    coords = collect_centroid_fov_3d(regular_spines, scan_angle=scan_angle)
    coords_BLA = collect_centroid_fov_3d(spines_BLA, scan_angle=scan_angle)
    coords_CA3 = collect_centroid_fov_3d(spines_CA3, scan_angle=scan_angle)

    if coords.size == 0 and coords_BLA.size == 0 and coords_CA3.size == 0:
        raise ValueError("No spine data found for animation.")

    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    x_BLA, y_BLA, z_BLA = coords_BLA[:, 0], coords_BLA[:, 1], coords_BLA[:, 2]
    x_CA3, y_CA3, z_CA3 = coords_CA3[:, 0], coords_CA3[:, 1], coords_CA3[:, 2]

    # Setup figure and 3D axis
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection='3d')

    scatter = ax.scatter(
        x,
        y if neuron_orientation == "horizontal" else z,
        z if neuron_orientation == "horizontal" else y,
        color=spine_color,
        s=spine_size,
        edgecolors=spine_edgecolor,
        alpha=0.4,
        zorder=0)
    scatter_BLA = ax.scatter(
        x_BLA,
        y_BLA if neuron_orientation == "horizontal" else z_BLA,
        z_BLA if neuron_orientation == "horizontal" else y_BLA,
        color=BLA_spine_color,
        s=BLA_spine_size,
        edgecolors=BLA_spine_edgecolor,
        alpha=1.,
        zorder=10)
    scatter_CA3 = ax.scatter(
        x_CA3,
        y_CA3 if neuron_orientation == "horizontal" else z_CA3,
        z_CA3 if neuron_orientation == "horizontal" else y_CA3,
        color=CA3_spine_color,
        s=CA3_spine_size,
        edgecolors=CA3_spine_edgecolor,
        alpha=1.,
        zorder=10)

    if not show_ticks:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_zticks([])
    
    ax.set(xticklabels=[], yticklabels=[], zticklabels=[])

    # Set axis limits for equal aspect ratio
    if axis_lims:
        ax.set(
            xlim=(axis_lims[0], axis_lims[1]),
            ylim=(axis_lims[2], axis_lims[3]),
            zlim=(axis_lims[4], axis_lims[5])
        )
    else:
        max_range = np.array([
            max(x) - min(x),
            max(y) - min(y),
            max(z) - min(z)
        ]).max() / 2.0

        mid_x = (max(x) + min(x)) * 0.5
        mid_y = (max(y) + min(y)) * 0.5
        mid_z = (max(z) + min(z)) * 0.5

        if neuron_orientation == "horizontal":
            ax.set(
                xlim=(mid_x - max_range, mid_x + max_range),
                ylim=(mid_y - max_range, mid_y + max_range),
                zlim=(mid_z - max_range, mid_z + max_range)
            )
        else:
            ax.set(
                xlim=(mid_x - max_range, mid_x + max_range),
                zlim=(mid_y - max_range, mid_y + max_range),
                ylim=(mid_z - max_range, mid_z + max_range)
            )

    # Update function for animation
    def update(frame):
        # current_elev = elev_start + (elev_end - elev_start) * (frame / (frames - 1))
        current_elev = 15
        current_azim = azim_start + (azim_end - azim_start) * (frame / (frames - 1))
        ax.view_init(elev=current_elev, azim=current_azim)
        return scatter, scatter_BLA, scatter_CA3

    # Create animation
    anim = FuncAnimation(fig, update, frames=frames, interval=interval, blit=False)

    # Save animation if path is provided
    if save_path:
        anim.save(save_path, writer="pillow")

    return anim


def make_event_kernel(
        length=10,
        tau=3
) -> np.ndarray:
    """
    Create an exponential decay kernel for event detection.
    The kernel is normalized to have zero mean and unit variance.
    Parameters
    ----------
    length : int, optional
        Length of the kernel in time points. Default is 10.
    tau : float, optional
        Time constant for the exponential decay. Default is 3.
    Returns
    -------
    np.ndarray
        Normalized kernel array of shape (length,).
    """

    t = np.arange(length)
    kernel = np.exp(-t / tau)
    kernel = (kernel - np.mean(kernel)) / np.std(kernel)
    
    return kernel


def plot_zscore_heatmap(
        all_spines_instance: object,
        spines: list,
        ax: plt.Axes,
        figsize: tuple,
        input_type: str,
        average: bool,
        sort: bool,
        sort_window: tuple,
        cmap: str,
        cmap_extent: list,
        fontsize: int,
        cbar_shrink: float,
        xlim: tuple,
        **kwargs,
) -> None:
    """
    Plot heatmaps of z-scores for all spines in the dataset.

    Parameters
    ----------
    all_spines : object
        Dataset object containing spine and dendrite data, including z-scores
        and timestamps for BLA and CA3 regions.
    spines : list
        List of spine indices to plot.
    ax : plt.Axes
        Matplotlib Axes object to plot on. If None, a new figure is created.
    figsize : tuple
        Figure size (width, height) in inches.
    input_type : str, optional
        Specifies the source of z-scores to plot. Valid options are:
        - "BLA" for z-scores of BLA spines.
        - "CA3" for z-scores of CA3 spines.
        - None (default) to plot both BLA and CA3 spines in separate subplots.
    average : bool, optional
        If True, plot the average z-score across sweeps for each spine.
        Default is False, which shows all sweeps.
    sort : bool, optional
        If True, sort spines based on the average z-score in the specified
        time window (`sort_window`). Default is False.
    sort_window : tuple, optional
        Time window (start, end) for calculating the sorting criterion, where
        the indices refer to the time points in the z-score traces.
        Default is (16, 21).
    cmap : str, optional
        Colormap for the heatmap. Default is 'viridis'.
    cmap_extent : list, optional
        Extent of the colormap as [vmin, vmax]. Default is None.
    fontsize : int, optional
        Font size for axis labels and ticks. Default is 14.
    cbar_shrink : float, optional
        Shrinkage factor for the colorbar. Default is 0.5.
    xlim : tuple, optional
        X-axis limits as (xmin, xmax). Default is None.

    Returns
    -------
    tuple or None
        - If `input_type` is None and `sort` is True, returns a tuple of sorted
          indices for BLA and CA3 spines:
          `(sorted_indices_BLA, sorted_indices_CA3)`.
        - If `input_type` is "BLA" or "CA3" and `sort` is True, returns the
          sorted indices for the specified input type.
        - If `sort` is False, returns None and directly plots the heatmap(s).

    Notes
    -----
    - This function generates heatmaps of z-scores over time for spines, with
      color intensity representing the magnitude of the z-scores.
    - Sorting is based on the mean z-score within the specified time window.
    - When `average` is True, each spine is represented by its averaged z-score
      trace. Otherwise, all sweeps are displayed.
    - If `input_type` is None, heatmaps for BLA and CA3 spines are displayed in
      separate subplots.
    """

    start, end = sort_window

    if cmap_extent:
        vmin, vmax = cmap_extent

    spine_indices = [
        n for n, spine in enumerate(all_spines_instance.spines_data)
        if spine in spines]

    if input_type is None:
        zscore_BLA = [
            spine for n, spine in enumerate(all_spines_instance.zscores_BLA)
            if n in spine_indices]
        ts_BLA = [
            ts for n, ts in enumerate(all_spines_instance.ts_BLA)
            if n in spine_indices]
        zscore_CA3 = [
            spine for n, spine in enumerate(all_spines_instance.zscores_CA3)
            if n in spine_indices]
        ts_CA3 = [
            ts for n, ts in enumerate(all_spines_instance.ts_CA3)
            if n in spine_indices]

        figsize = (figsize[0] * 2, figsize[1])

        if average:
            processed_zscores_BLA = np.array(
                [np.mean(spine_sweeps, axis=0)
                 for spine_sweeps in zscore_BLA])
            processed_timestamps_BLA = np.array(ts_BLA)

            processed_zscores_CA3 = np.array(
                [np.mean(spine_sweeps, axis=0)
                 for spine_sweeps in zscore_CA3])
            processed_timestamps_CA3 = np.array(ts_CA3)
            y_label = "Sorted spines" if sort else "Spines"

        else:
            processed_zscores_BLA = np.concatenate(zscore_BLA, axis=0)
            processed_timestamps_BLA = np.concatenate(
                [np.array(ts) for ts in ts_BLA], axis=0)

            processed_zscores_CA3 = np.concatenate(zscore_CA3, axis=0)
            processed_timestamps_CA3 = np.concatenate(
                [np.array(ts) for ts in ts_CA3], axis=0)
            y_label = (
                f"Sorted sweeps of"
                f"{len(all_spines_instance.spines_data)} spines"
                if sort else
                f"Sweeps of"
                f"{len(all_spines_instance.spines_data)} spines")

        if sort:
            if not sort_window:
                raise AttributeError("sort_window not defined.")
            
            kernel = make_event_kernel(length=end - start)

            sorting_values_BLA = np.array([
                np.max(correlate(trace[start:end], kernel, mode='valid'))
                for trace in processed_zscores_BLA
            ])

            # sorting_values_BLA = np.array(
            #     [np.sum(trace[start:end])
            #      for trace in processed_zscores_BLA])
            
            sorted_indices_BLA = np.argsort(sorting_values_BLA)
            processed_zscores_BLA = processed_zscores_BLA[sorted_indices_BLA]
            processed_timestamps_BLA = processed_timestamps_BLA[
                sorted_indices_BLA]
            
            sorting_values_CA3 = np.array([
                np.max(correlate(trace[start:end], kernel, mode='valid'))
                for trace in processed_zscores_CA3
            ])

            # sorting_values_CA3 = np.array(
            #     [np.sum(trace[start:end])
            #      for trace in processed_zscores_CA3])

            sorted_indices_CA3 = np.argsort(sorting_values_CA3)
            processed_zscores_CA3 = processed_zscores_CA3[sorted_indices_CA3]
            processed_timestamps_CA3 = processed_timestamps_CA3[
                sorted_indices_CA3]

        if ax and not isinstance(ax, np.ndarray) and not ax.shape == (1, 2):
            raise ValueError("Invalid ax shape. Must be (1, 2).")

        if not ax:
            _, ax_zscore = plt.subplots(
                1, 2,
                figsize=figsize,
                sharex=True,
                sharey=True,
                layout="constrained")

        if cmap_extent is None:
            vmin_BLA = np.min(processed_zscores_BLA)
            vmax_BLA = np.max(processed_zscores_BLA)

            vmin_CA3 = np.min(processed_zscores_BLA)
            vmax_CA3 = np.max(processed_zscores_BLA)

        else:
            vmin_BLA, vmax_BLA = vmin, vmax
            vmin_CA3, vmax_CA3 = vmin, vmax

        BLA = ax_zscore[0].imshow(
            processed_zscores_BLA,
            aspect='auto',
            cmap=cmap,
            extent=[
                np.min(processed_timestamps_BLA),
                np.max(processed_timestamps_BLA),
                0,
                processed_zscores_BLA.shape[0]],
            interpolation='none',
            vmin=vmin_BLA,
            vmax=vmax_BLA)

        CA3 = ax_zscore[1].imshow(
            processed_zscores_CA3,
            aspect='auto',
            cmap=cmap,
            extent=[
                np.min(processed_timestamps_CA3),
                np.max(processed_timestamps_CA3),
                0,
                processed_zscores_CA3.shape[0]],
            interpolation='none',
            vmin=vmin_CA3,
            vmax=vmax_CA3)

        for ax in ax_zscore:
            ax.axvline(
                x=1.0,
                ymin=1,
                ymax=1.01,
                color='black',
                lw=1,
                clip_on=False)

            if average:
                ax.set_yticks([0, len(processed_zscores_BLA)])

        divider = make_axes_locatable(ax_zscore[1])
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(CA3, cax=cax, shrink=cbar_shrink)
        cbar.ax.set_ylabel('Z-score', fontsize=fontsize, rotation=90)
        cbar.ax.tick_params(labelsize=fontsize)

        ax_zscore[0].set_xlabel('Time (s)', fontsize=fontsize)
        ax_zscore[0].set_ylabel(y_label, fontsize=fontsize, labelpad=-20)
        ax_zscore[0].tick_params("both", labelsize=fontsize)
        ax_zscore[1].tick_params("both", labelsize=fontsize)
        ax_zscore[0].set_yticks([0, len(processed_zscores_BLA)])
        ax_zscore[0].invert_yaxis()

        if xlim:
            ax_zscore[0].set_xlim(xlim)
            ax_zscore[1].set_xlim(xlim)

    else:
        if input_type == 'BLA':
            selected_zscore = all_spines_instance.zscores_BLA
            selected_ts = all_spines_instance.ts_BLA
        elif input_type == 'CA3':
            selected_zscore = all_spines_instance.zscores_CA3
            selected_ts = all_spines_instance.ts_CA3
        else:
            raise ValueError(
                "Invalid input_type. Must be 'BLA', 'CA3', or None.")

        zscore = [
                spine for n, spine in enumerate(selected_zscore)
                if n in spine_indices]
        ts = [
            ts for n, ts in enumerate(selected_ts)
            if n in spine_indices]

        if average:
            processed_zscores = np.array(
                [np.mean(spine_sweeps, axis=0)
                 for spine_sweeps in zscore])
            processed_timestamps = np.array(ts)
            y_label = "Sorted spines" if sort else "Spines"
        else:
            processed_zscores = np.concatenate(zscore, axis=0)
            processed_timestamps = np.concatenate(
                [np.array(t) for t in ts], axis=0)
            y_label = (
                f"Sorted sweeps of"
                f"{len(all_spines_instance.spines_data)} spines"
                if sort else
                f"Sweeps of"
                f"{len(all_spines_instance.spines_data)} spines")

        if sort:
            kernel = make_event_kernel(length=end - start)

            sorting_values = np.array([
                np.max(correlate(trace[start:end], kernel, mode='valid'))
                for trace in processed_zscores
            ])

            # sorting_values = np.array(
            #     [np.sum(trace[start:end])
            #      for trace in processed_zscores])
            
            sorted_indices = np.argsort(sorting_values)
            processed_zscores = processed_zscores[sorted_indices]
            processed_timestamps = processed_timestamps[sorted_indices]

        if not ax:
            _, ax = plt.subplots(
                1, 1,
                figsize=figsize,
                layout="constrained")

        if cmap_extent is None:
            vmin = np.min(processed_zscores)
            vmax = np.max(processed_zscores)

        img = ax.imshow(
            processed_zscores,
            aspect='auto',
            cmap=cmap,
            extent=[np.min(processed_timestamps),
                    np.max(processed_timestamps),
                    0,
                    processed_zscores.shape[0]],
            interpolation='none',
            vmin=vmin,
            vmax=vmax)

        ax.axvline(
            x=1.0,
            ymin=1,
            ymax=1.01,
            color='black',
            lw=1,
            clip_on=False)

        if average:
            ax.set_yticks([0, len(processed_zscores)])

        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)

        cbar = plt.colorbar(img, cax=cax, shrink=cbar_shrink)
        cbar.ax.set_ylabel('Z-score', fontsize=fontsize)
        cbar.ax.tick_params(labelsize=fontsize)

        ax.set_xlabel('Time (s)', fontsize=fontsize)
        ax.set_ylabel(y_label, fontsize=fontsize, labelpad=-20)
        ax.tick_params("both", labelsize=fontsize)
        ax.set_yticks([0, len(processed_zscores)])
        ax.invert_yaxis()

        if xlim:
            ax.set_xlim(xlim)


def plot_events_spines(
        spines: list,
        events: list,
        n_event_threshold: int,
        spine_size: int,
        spine_edgecolor: int,
        fontsize: int,
        ax: plt.Axes,
        show_cmap: bool,
        cbar_width: float,
        cmap: str,
        scan_angle: bool,
        zorder: int,
        **scatter_kwargs,
) -> None:

    if not isinstance(spines, list):
        raise TypeError("spines must be a list of dictionaries.")

    if n_event_threshold > len(events[0]):
        raise ValueError(
            f"n_event_threshold should be <= {len(events[0])}")

    all_centroids = np.array(
        collect_centroid_fov(spines, scan_angle=scan_angle))

    if len(all_centroids) == 0:
        return

    event_counts = [sum(flag) for flag in events]

    # Filter out spines with n events
    nonzero_centroids = [
        centroid for centroid, count in zip(all_centroids, event_counts)
        if count > n_event_threshold]
    nonzero_event_counts = [
        count for count in event_counts
        if count > n_event_threshold]

    if len(nonzero_centroids) == 0:
        return

    if ax is None:
        _, ax = plt.subplots(
            figsize=(8, 8),
            layout="constrained")

    ax.set_xlabel('X coordinate', fontsize=fontsize)
    ax.set_ylabel('Y coordinate', fontsize=fontsize)
    ax.set_aspect('equal')
    ax.tick_params("both", labelsize=fontsize)

    scatter = ax.scatter(
        [c[0] for c in nonzero_centroids],  # X coordinates of selected spines
        [c[1] for c in nonzero_centroids],  # Y coordinates of selected spines
        s=spine_size,
        c=nonzero_event_counts,
        edgecolors=spine_edgecolor,
        cmap=cmap,
        vmin=1,
        vmax=5,
        zorder=zorder,
        **scatter_kwargs
    )

    if show_cmap:
        cbar = plt.colorbar(
            scatter,
            ax=ax,
            label='Number of Events',
            shrink=cbar_width)

        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label('Number of Events', fontsize=fontsize)
        cbar.set_ticks([1, 2, 3, 4, 5])


def spine_sholl(
        morphology: object,
        radius_step: float,
        n_radii: int,
        spines: list,
        events: list,
        n_event_threshold: int,
        show_plot: bool,
        show_sholl_curve: bool,
        ax: plt.Axes,
        ax_sholl_curve: plt.Axes,
        circle_color: str ,
        circle_linestyle: str,
        circle_linewidth: int or float,
        countline_color: str,
        countline_width: int or float,
        countline_style: str,
        countline_alpha: float,
        size: int or float,
        fontsize: int,
        cmap: str,
        colorbar_orientation: str,
        scan_angle: bool
) -> tuple:

    # Extract the center point (soma) coordinates
    soma = morphology.soma
    obj_resolution = morphology.objective_resolution
    center_point = (soma.x, soma.y, soma.z)

    # Get spine centroids and event counts
    all_centroids = np.array(collect_centroid_fov(spines, scan_angle))

    if events:
        event_counts = [sum(flags) for flags in events]

        # Filter out spines with 0 events
        nonzero_centroids = [
            centroid for centroid, count in zip(
                all_centroids, event_counts)
            if count > n_event_threshold]

        nonzero_event_counts = [
            count
            for count in event_counts
            if count > n_event_threshold]

        spine_positions = np.copy(nonzero_centroids)
        spine_positions[:, 0:2] *= obj_resolution  # convert to µm

    else:
        spine_positions = np.copy(all_centroids)
        spine_positions[:, 0:2] *= obj_resolution  # convert to µm

    max_radius = radius_step * n_radii
    radii = np.arange(0, max_radius + radius_step, radius_step)

    # Initialize counts for apical and basal dendrites
    apical_counts = []
    basal_counts = []

    # Loop over each radius
    for i, radius in enumerate(radii):
        distances = np.sqrt(
            (spine_positions[:, 0] - center_point[0]) ** 2 +
            (spine_positions[:, 1] - center_point[1]) ** 2 +
            (spine_positions[:, 2] - center_point[2]) ** 2)

        if i == 0:
            # For the first radius, count spines within the first circle
            spine_indices_in_shell = distances <= radius
        else:
            # For subsequent radii, count spines within the shell between radii
            previous_radius = radii[i - 1]
            spine_indices_in_shell = ((distances <= radius) &
                                      (distances > previous_radius))

        # Initialize counts for the current shell
        apical_count = 0
        basal_count = 0

        # Loop through spines in the current shell
        for idx in np.where(spine_indices_in_shell)[0]:
            spine_position = spine_positions[idx]

            # Find the closest node in morphology
            closest_node = min(
                morphology.neuron,  # List of nodes
                key=lambda node: np.sqrt(
                    (spine_position[0] - node.x) ** 2 +
                    (spine_position[1] - node.y) ** 2 +
                    (spine_position[2] - node.z) ** 2))

            # Increment the count based on node type (apical or basal)
            if closest_node.type == 'apical dendrite':
                apical_count += 1
            elif closest_node.type == 'basal dendrite':
                basal_count += 1

        # Append the counts for this radius
        apical_counts.append(apical_count)
        basal_counts.append(basal_count)

    # Plotting the spines and Sholl circles
    if not show_plot and not show_sholl_curve:
        return np.array(apical_counts), np.array(basal_counts), radii
    
    if show_plot:
        if ax is None:
            _, ax = plt.subplots(
                figsize=(8, 8),
                layout="constrained")

        for radius in radii:
            circle = plt.Circle(
                (center_point[0], center_point[1]),
                radius,
                color=circle_color, fill=False,
                linestyle=circle_linestyle,
                linewidth=circle_linewidth)
            ax.add_artist(circle)

        if events:
            scatter = ax.scatter(
                [c[0] for c in spine_positions],
                [c[1] for c in spine_positions],
                c=nonzero_event_counts,
                vmin=1,  # Minimum value for the color scale (at least 1 event)
                vmax=5,  # Maximum value for the color scale (up to 5 events)
                s=size,
                cmap=cmap
            )
            cbar = plt.colorbar(scatter, ax=ax, orientation=colorbar_orientation)
            cbar.set_label('Number of Events', fontsize=fontsize)
            cbar.ax.tick_params(labelsize=fontsize)

        else:
            ax.scatter(
                [c[0] for c in spine_positions],
                [c[1] for c in spine_positions],
                c='gray',
                s=size,
            )
        ax.set_xlabel('X coordinate', fontsize=fontsize)
        ax.set_ylabel('Y coordinate', fontsize=fontsize)
        ax.set_aspect('equal', adjustable='box')

    # Plot the Sholl curve (apical and basal separately)
    if show_sholl_curve:
        if not ax_sholl_curve:
            _, ax_sholl_curve = plt.subplots(
                figsize=(6, 4),
                layout="constrained")

        ax_sholl_curve.plot(
            radii,
            apical_counts,
            color=countline_color,
            lw=countline_width,
            ls=countline_style,
            alpha=countline_alpha,
            clip_on=False)

        ax_sholl_curve.plot(
            -radii,
            basal_counts,
            color=countline_color,
            lw=countline_width,
            ls=countline_style,
            alpha=countline_alpha,
            clip_on=False)

        ax_sholl_curve.set_xlabel('Radius (μm)', fontsize=fontsize)
        ax_sholl_curve.set_ylabel('Number of Spines', fontsize=fontsize)

    return np.array(apical_counts), np.array(basal_counts), radii
