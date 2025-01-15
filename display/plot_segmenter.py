""" Created on Mon Jun  3 15:02:25 2024
    @author: dcupolillo """

import numpy as np
import matplotlib.pyplot as plt


def collect_centroid_fov(
        data: list[dict]
) -> list[tuple[float, float]]:
    """
    Collects all 'centroid_fov' tuples from a 2D list of dictionaries
    into a 1D list.

    Parameters
    ----------
    data : List[List[dict]]
        A 2D list of dictionaries containing 'centroid_fov' keys.

    Returns
    -------
    List[Tuple[float, float]]
        A 1D list of 'centroid_fov' tuples.
    """
    centroid_fov_list = []
    for spine in data:
        centroid_fov_list.append(tuple(
            spine['centroid_fov'] + [spine['roi_z']]))
    return centroid_fov_list


def plot_all_spines(
        all_spines: list,
        spine_size: int,
        fontsize: int,
        spine_color: str or tuple[float, float, float],
        ax: plt.Axes = None,
) -> None:
    """
    Plot all spine centroids from the segmenter.

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

    Raises
    ------
    ValueError
        If no spine data is found in the segmenter.

    Returns
    -------
    None
    """

    # Convert CuPy array to NumPy array for plotting
    all_centroids = np.array(collect_centroid_fov(all_spines))

    if len(all_centroids) == 0:
        raise ValueError("No spine data found in the segmenter.")

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

        # Set labels and colors
        ax.set_xlabel('X coordinate', fontsize=fontsize)
        ax.set_ylabel('Y coordinate', fontsize=fontsize)

        # Set equal aspect ratio
        ax.set_aspect('equal', adjustable='box')

    ax.scatter(all_centroids[:, 0],
               all_centroids[:, 1],
               s=spine_size,
               color=spine_color)


# def plot_all_dendrites(
#         all_dendrites: np.ndarray,
#         dendrite_size: int,
#         dendrite_color: str or tuple[float, float, float],
#         ax: plt.Axes = None,
# ) -> None:
#     """
#     Plot all spine centroids from the segmenter.

#     Parameters
#     ----------
#     all_spines : cp.ndarray
#         The spine data as a CuPy array.
#     spine_size : int, optional
#         Size of the scatter points, by default 5.
#     spine_color : str or tuple, optional
#         Color of the scatter points, by default 'cyan'.
#     ax : matplotlib.axes.Axes, optional
#         Matplotlib Axes object to plot on. If None, a new figure is created.

#     Raises
#     ------
#     ValueError
#         If no spine data is found in the segmenter.

#     Returns
#     -------
#     None
#     """
#     # Convert CuPy array to NumPy array for plotting
#     print("Converting CuPy array to NumPy array for plotting...")
#     all_dendrites_np = all_dendrites.get()

#     # Plot with Matplotlib
#     print("Plotting data...")
#     if ax is None:
#         fig, ax = plt.subplots(figsize=(8, 8))

#     ax.scatter([x[0] for x in all_dendrites_np],
#                [y[1] for y in all_dendrites_np],
#                s=dendrite_size,
#                color=dendrite_color)

#     # Set equal aspect ratio
#     ax.set_aspect('equal', adjustable='box')

#     # Set labels and colors
#     ax.set_xlabel('X coordinate')
#     ax.set_ylabel('Y coordinate')

#     print("Plotting completed.")


def plot_all_zscore_heatmap(
        all_spines: object,
        input_type: str,
        average: bool,
        sort: bool):

    if not input_type:
        zscore_BLA = all_spines.batch_z_scores_BLA
        ts_BLA = all_spines.batch_ts_BLA
        zscore_CA3 = all_spines.batch_z_scores_CA3
        ts_CA3 = all_spines.batch_ts_CA3

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
            y_label = f"Sweeps of {len(all_spines.batch_spines_data)} spines"

        if sort:
            sorting_values_BLA = np.array(
                [np.mean(trace[16:21]) for trace in processed_zscores_BLA])
            sorted_indices_BLA = np.argsort(sorting_values_BLA)
            processed_zscores_BLA = processed_zscores_BLA[sorted_indices_BLA]
            processed_timestamps_BLA = processed_timestamps_BLA[
                sorted_indices_BLA]

            sorting_values_CA3 = np.array(
                [np.mean(trace[16:21]) for trace in processed_zscores_CA3])
            sorted_indices_CA3 = np.argsort(sorting_values_CA3)
            processed_zscores_CA3 = processed_zscores_CA3[sorted_indices_CA3]
            processed_timestamps_CA3 = processed_timestamps_CA3[
                sorted_indices_CA3]

        fig_zcore, ax_zscore = plt.subplots(1, 2, sharex=True, sharey=True)

        BLA = ax_zscore[0].imshow(
            processed_zscores_BLA,
            aspect='auto',
            cmap='viridis',
            extent=[np.min(processed_timestamps_BLA),
                    np.max(processed_timestamps_BLA),
                    0,
                    processed_zscores_BLA.shape[0]],
            interpolation='none')

        CA3 = ax_zscore[1].imshow(
            processed_zscores_CA3,
            aspect='auto',
            cmap='viridis',
            extent=[np.min(processed_timestamps_CA3),
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

        plt.colorbar(BLA, ax=ax_zscore[0], label='Z-score')
        plt.colorbar(CA3, ax=ax_zscore[1], label='Z-score')
        ax_zscore[0].set_xlabel('Time (s)')
        ax_zscore[0].set_ylabel(y_label)
        ax_zscore[0].invert_yaxis()

        if sort:
            return sorted_indices_BLA, sorted_indices_CA3

    else:
        if input_type == 'BLA':
            zscore = all_spines.batch_z_scores_BLA
            ts = all_spines.batch_ts_BLA
        elif input_type == 'CA3':
            zscore = all_spines.batch_z_scores_CA3
            ts = all_spines.batch_ts_CA3
        else:
            raise ValueError(
                "Invalid input_type. Must be 'BLA', 'CA3', or None.")

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
            y_label = f"Sweeps of {len(all_spines.batch_spines_data)} spines"

        if sort:
            sorting_values = np.array(
                [np.mean(trace[16:21]) for trace in processed_zscores])
            sorted_indices = np.argsort(sorting_values)
            processed_zscores = processed_zscores[sorted_indices]
            processed_timestamps = processed_timestamps[sorted_indices]

        fig, ax = plt.subplots(1, 1)
        img = ax.imshow(
            processed_zscores,
            aspect='auto',
            cmap='viridis',
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

        plt.colorbar(img, ax=ax, label='Z-score')
        ax.set_xlabel('Time (s)')
        ax.set_ylabel(y_label)
        ax.invert_yaxis()

    # return processed_zscores_BLA


def plot_events_spines(
        all_spines: list,
        events_probability: list,
        n_event_threshold: int,
        spine_size: int,
        fontsize: int,
        ax: plt.Axes = None,
        show_cmap: bool = None,
        cmap: str = 'viridis',
        dynamic_threshold_percentile: int = None,
) -> None:

    if n_event_threshold > len(events_probability[0]):
        raise ValueError(
            f"n_event_threshold should be <= {len(events_probability[0])}")

    all_centroids = np.array(collect_centroid_fov(all_spines))

    if len(all_centroids) == 0:
        raise ValueError("No spine data found.")

    flatten_probabilities = np.array(
        [sweep for spine in events_probability for sweep in spine]
    )

    dynamic_threshold = np.percentile(
        flatten_probabilities[~np.isnan(flatten_probabilities)],
        dynamic_threshold_percentile)

    binary_flag = np.zeros_like(events_probability)
    for n, spine in enumerate(events_probability):
        for i, sweep_prob in enumerate(spine):
            if sweep_prob >= dynamic_threshold:
                binary_flag[n][i] = 1

    event_counts = [sum(flag) for flag in binary_flag]

    # Filter out spines with n events
    nonzero_centroids = [
        centroid for centroid, count in zip(all_centroids, event_counts)
        if count > n_event_threshold]
    nonzero_event_counts = [
        count for count in event_counts
        if count > n_event_threshold]

    if len(nonzero_centroids) == 0:
        raise ValueError("No spines with events found.")

    # Plot with Matplotlib
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    ax.set_xlabel('X coordinate', fontsize=fontsize)
    ax.set_ylabel('Y coordinate', fontsize=fontsize)
    ax.set_aspect('equal', adjustable='box')

    scatter = ax.scatter(
        [c[0] for c in nonzero_centroids],  # X coordinates of filtered spines
        [c[1] for c in nonzero_centroids],  # Y coordinates of filtered spines
        s=spine_size,
        c=nonzero_event_counts,
        cmap=cmap,
        vmin=1,  # Minimum value for the color scale (at least 1 event)
        vmax=5   # Maximum value for the color scale (up to 5 events)
    )

    if show_cmap:
        cbar = plt.colorbar(
            scatter, ax=ax, label='Number of Events', shrink=0.8)

        cbar.ax.tick_params(labelsize=fontsize)
        cbar.set_label('Number of Events', fontsize=fontsize)
        cbar.set_ticks([1, 2, 3, 4, 5])


def spine_sholl(
        all_spines: dict,
        morphology: object,
        events_flag: list,
        n_event_threshold: int,
        radius_step: float,
        n_radii: int,
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
    all_centroids = np.array(collect_centroid_fov(all_spines))

    if events_flag:
        event_counts = [sum(flags) for flags in events_flag]

        # Filter out spines with 0 events
        nonzero_centroids = [
            centroid for centroid, count in zip(all_centroids, event_counts)
            if count > n_event_threshold]
        nonzero_event_counts = [count
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
        distances = np.sqrt((spine_positions[:, 0] - center_point[0]) ** 2 +
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
                key=lambda node: np.sqrt((spine_position[0] - node.x) ** 2 +
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
        fig, ax = plt.subplots(figsize=(8, 8))

    for radius in radii:
        circle = plt.Circle(
            (center_point[0], center_point[1]),
            radius,
            color=circle_color, fill=False,
            linestyle=circle_linestyle,
            linewidth=circle_linewidth)
        ax.add_artist(circle)

    if events_flag:
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
        fig_sholl_curve, ax_sholl_curve = plt.subplots(figsize=(6, 4))

    ax_sholl_curve.plot(radii, apical_counts,
                        color=countline_color,
                        lw=countline_width,
                        ls=countline_style,
                        alpha=countline_alpha,
                        clip_on=False)
    ax_sholl_curve.plot(-radii, basal_counts,
                        color=countline_color,
                        lw=countline_width,
                        ls=countline_style,
                        alpha=countline_alpha,
                        clip_on=False)
    ax_sholl_curve.set_xlabel('Radius (μm)', fontsize=fontsize)
    ax_sholl_curve.set_ylabel('Number of Spines', fontsize=fontsize)

    return np.array(apical_counts), np.array(basal_counts), radii
