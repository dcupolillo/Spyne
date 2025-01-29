""" Created on Mon Jun  3 15:02:25 2024
    @author: dcupolillo """

import numpy as np
import matplotlib.pyplot as plt


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


def plot_spines_2d(
        spines: list,
        spine_size: int,
        fontsize: int,
        spine_color: str,
        spine_edgecolor: str,
        ax: plt.Axes,
        scan_angle: bool,
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
            layout="constrained")

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
        edgecolors=spine_edgecolor)

    plt.show()


def plot_zscore_heatmap(
        all_spines_instance: object,
        spines: list,
        input_type: str,
        average: bool,
        sort: bool,
        sort_window: tuple,
        cmap: str,
) -> None:
    """
    Plot heatmaps of z-scores for all spines in the dataset.

    Parameters
    ----------
    all_spines : object
        Dataset object containing spine and dendrite data, including z-scores 
        and timestamps for BLA and CA3 regions.
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

    Returns
    -------
    tuple or None
        - If `input_type` is None and `sort` is True, returns a tuple of sorted
          indices for BLA and CA3 spines: `(sorted_indices_BLA, sorted_indices_CA3)`.
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

    spine_indices = [
        n for n, spine in enumerate(all_spines_instance.spines_data)
        if spine in spines]
    
    if not input_type:
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

        if average:
            processed_zscores_BLA = np.array(
                [np.mean(spine_sweeps, axis=0)
                 for spine_sweeps in zscore_BLA])
            processed_timestamps_BLA = ts_BLA

            processed_zscores_CA3 = np.array(
                [np.mean(spine_sweeps, axis=0)
                 for spine_sweeps in zscore_CA3])
            processed_timestamps_CA3 = ts_CA3
            y_label = "Spines"

        else:
            processed_zscores_BLA = np.concatenate(zscore_BLA, axis=0)
            processed_timestamps_BLA = np.concatenate(
                [np.array(ts) for ts in ts_BLA], axis=0)

            processed_zscores_CA3 = np.concatenate(zscore_CA3, axis=0)
            processed_timestamps_CA3 = np.concatenate(
                [np.array(ts) for ts in ts_CA3], axis=0)
            y_label = f"Sweeps of {len(all_spines_instance.spines_data)} spines"

        if sort:
            if not sort_window:
                raise AttributeError("sort_window not defined.")

            sorting_values_BLA = np.array(
                [np.mean(trace[start:end]) for trace in processed_zscores_BLA])
            sorted_indices_BLA = np.argsort(sorting_values_BLA)
            processed_zscores_BLA = processed_zscores_BLA[sorted_indices_BLA]
            processed_timestamps_BLA = processed_timestamps_BLA[
                sorted_indices_BLA]

            sorting_values_CA3 = np.array(
                [np.mean(trace[start:end]) for trace in processed_zscores_CA3])
            sorted_indices_CA3 = np.argsort(sorting_values_CA3)
            processed_zscores_CA3 = processed_zscores_CA3[sorted_indices_CA3]
            processed_timestamps_CA3 = processed_timestamps_CA3[
                sorted_indices_CA3]

        _, ax_zscore = plt.subplots(
            1, 2,
            sharex=True,
            sharey=True,
            layout="constrained")

        BLA = ax_zscore[0].imshow(
            processed_zscores_BLA,
            aspect='auto',
            cmap=cmap,
            extent=[
                np.min(processed_timestamps_BLA),
                np.max(processed_timestamps_BLA),
                0,
                processed_zscores_BLA.shape[0]],
            interpolation='none')

        CA3 = ax_zscore[1].imshow(
            processed_zscores_CA3,
            aspect='auto',
            cmap=cmap,
            extent=[
                np.min(processed_timestamps_CA3),
                np.max(processed_timestamps_CA3),
                0,
                processed_zscores_CA3.shape[0]],
            interpolation='none')

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

        plt.colorbar(CA3, ax=ax_zscore[1], label='Z-score')
        ax_zscore[0].set_xlabel('Time (s)')
        ax_zscore[0].set_ylabel(y_label)
        ax_zscore[0].invert_yaxis()

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
            processed_timestamps = ts
            y_label = "Spines"
        else:
            processed_zscores = np.concatenate(zscore, axis=0)
            processed_timestamps = np.concatenate(
                [np.array(t) for t in ts], axis=0)
            y_label = f"Sweeps of {len(all_spines_instance.spines_data)} spines"

        if sort:
            sorting_values = np.array(
                [np.mean(trace[start:end])
                 for trace in processed_zscores])
            sorted_indices = np.argsort(sorting_values)
            processed_zscores = processed_zscores[sorted_indices]
            processed_timestamps = processed_timestamps[sorted_indices]

        _, ax = plt.subplots(
            1, 1,
            layout="constrained")

        img = ax.imshow(
            processed_zscores,
            aspect='auto',
            cmap=cmap,
            extent=[np.min(processed_timestamps),
                    np.max(processed_timestamps),
                    0,
                    processed_zscores.shape[0]],
            interpolation='none')

        ax.axvline(
            x=1.0,
            ymin=1,
            ymax=1.01,
            color='black',
            lw=1,
            clip_on=False)

        if average:
            ax.set_yticks([0, len(processed_zscores)])

        plt.colorbar(img, ax=ax, label='Z-score')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(y_label)
        ax.invert_yaxis()

    plt.show()


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
        zorder: int
) -> None:

    if not isinstance(spines, list):
        raise TypeError("spines must be a list of dictionaries.")

    if n_event_threshold > len(events[0]):
        raise ValueError(
            f"n_event_threshold should be <= {len(events[0])}")

    all_centroids = np.array(
        collect_centroid_fov(spines, scan_angle=scan_angle))

    if len(all_centroids) == 0:
        raise ValueError("No spine data found.")

    event_counts = [sum(flag) for flag in events]

    # Filter out spines with n events
    nonzero_centroids = [
        centroid for centroid, count in zip(all_centroids, event_counts)
        if count > n_event_threshold]
    nonzero_event_counts = [
        count for count in event_counts
        if count > n_event_threshold]

    if len(nonzero_centroids) == 0:
        raise ValueError("No spines with events found.")

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
        vmin=1,  # Minimum value for the color scale (at least 1 event)
        vmax=5,  # Maximum value for the color scale (up to 5 events)
        zorder=zorder
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
        ax: plt.Axes = None,
        ax_sholl_curve: plt.Axes = None,
        circle_color: str = None,
        circle_linestyle: str = None,
        circle_linewidth: int or float = None,
        countline_color: str = None,
        countline_width: int or float = None,
        countline_style: str = None,
        countline_alpha: float = 0.5,
        size: int or float = None,
        fontsize: int = None,
        cmap: str = None,
        colorbar_orientation: str = None,):

    # Extract the center point (soma) coordinates
    soma = morphology.soma
    obj_resolution = morphology.objective_resolution
    center_point = (soma.x, soma.y, soma.z)

    # Get spine centroids and event counts
    all_centroids = np.array(collect_centroid_fov(spines))

    if events:
        event_counts = [sum(flags) for flags in events]

        # Filter out spines with 0 events
        nonzero_centroids = [
            centroid for centroid, count in zip(all_centroids, event_counts)
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
