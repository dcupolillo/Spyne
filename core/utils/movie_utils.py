""" Created on Tue Jan 23 11:02:59 2024
    @author: dcupolillo """

import numpy as np
import matplotlib.pyplot as plt


def modify_frames(
        frames: np.ndarray,
        deviating_rows_dict: dict
) -> np.ndarray:
    """
    Apply interpolation by averaging neighboring frames
    to the frames showing deviating rows.

    Parameters
    ----------
    frames : np.ndarray
        4D array of shape (n_frames, n_channels, height, width)
        representing the image data.
    deviating_rows_dict : dict
        Mapping from frame indices to lists of deviating row indices.

    Returns
    -------
    np.ndarray
        Modified frames with interpolated values for deviating rows.
    """
    modified_frames = np.copy(frames)

    for frame_number, deviating_rows in deviating_rows_dict.items():

        preceding_frame = max(frame_number - 1, 0)
        following_frame = min(frame_number + 1, len(frames) - 1)

        for row in deviating_rows:
            for channel in [0, 1]:
                average_row = np.mean(
                    [frames[preceding_frame, channel, row, :],
                     frames[following_frame, channel, row, :]],
                    axis=0
                )

                modified_frames[frame_number, channel, row, :] = average_row

    return modified_frames


def identify_rows_deviation(
        frames: np.ndarray,
        threshold_factor: int = 3,
        plot: bool = False
) -> dict:
    """
    Identify rows with significant intensity deviations across frames.

    Parameters
    ----------
    frames : np.ndarray
        4D array of shape (n_frames, n_channels, height, width)
        representing the image data.
    threshold_factor : int
        Factor to determine the sensitivity of deviation detection.
    plot : bool
        Whether to plot the intensity profiles and deviations.

    Returns
    -------
    dict
        Mapping from frame indices to lists of deviating row indices.
    """

    frame_to_rows = {}
    intensity_profiles = [np.mean(frame, axis=1) for frame in frames]
    average_profile = np.mean(intensity_profiles, axis=0)
    std_profile = np.std(intensity_profiles, axis=0)

    upper_tolerance_line = average_profile + threshold_factor * std_profile
    lower_tolerance_line = average_profile - threshold_factor * std_profile

    if plot:
        fig, axs = plt.subplots(
            1, 3,
            figsize=(10, 10),
            sharey=True,
            gridspec_kw={'width_ratios': [1, 1, 4]})

        cmap = plt.get_cmap('viridis')
        colors = [cmap(i / len(intensity_profiles))
                  for i in range(len(intensity_profiles))]

        deviation_detected = False
        deviating_frames = []  # Store frames with deviations

    for i, profile in enumerate(intensity_profiles):
        deviating_rows = []

        for j in range(len(profile)):
            if (profile[j] > upper_tolerance_line[j] or
                    profile[j] < lower_tolerance_line[j]):
                deviating_rows.append(j)

        deviating_rows.sort()
        consecutive_groups = []
        current_group = [deviating_rows[0]] if deviating_rows else []

        for k in range(1, len(deviating_rows)):
            if deviating_rows[k] == deviating_rows[k-1] + 1:
                current_group.append(deviating_rows[k])
            else:
                if len(current_group) > 1:
                    consecutive_groups.extend(current_group)
                current_group = [deviating_rows[k]]

        if len(current_group) > 1:
            consecutive_groups.extend(current_group)

        if consecutive_groups:
            frame_to_rows[i] = consecutive_groups
            if plot:
                deviating_frames.append(i)

        if plot:
            axs[-1].plot(
                profile,
                np.arange(len(profile)),
                color=colors[i],
                linewidth=1)

            if consecutive_groups:
                for row in consecutive_groups:
                    if row < len(profile) - 1:  # Avoid out-of-bounds error
                        axs[-1].plot(
                            [profile[row], profile[row+1]], [row, row+1],
                            color='red',
                            linewidth=3)
                deviation_detected = True

    if plot:
        if deviation_detected:
            for frame in deviating_frames:
                axs[-1].plot([], [],
                             color='red', linewidth=3, label=f'Frame {frame}')

        axs[-1].plot(
            upper_tolerance_line, np.arange(len(average_profile)), 'k--')
        axs[-1].plot(
            lower_tolerance_line, np.arange(len(average_profile)), 'k--')

        # Show one of the frames in grayscale
        n_frame = 15, 16
        axs[0].imshow(frames[n_frame[0]], cmap='gray', aspect='auto')
        axs[0].set_title(f"Frame {n_frame[0]}")
        axs[1].imshow(frames[n_frame[1]], cmap='gray', aspect='auto')
        axs[1].set_title(f"Frame {n_frame[1]}")

        # Highlight rows with deviations on the image
        for row in set([r for rows in frame_to_rows.values() for r in rows]):
            axs[0].axhspan(row - 0.5, row + 0.5, color='blue', alpha=0.3)

        # Set plot titles and labels
        axs[-1].set_title("Intensity Profile Across Frames")
        axs[0].set_ylabel("Row")
        axs[0].set_xlabel("Column")
        axs[-1].set_xlabel("Mean Intensity")
        axs[-1].invert_yaxis()
        axs[-1].grid(True)

        axs[-1].legend(loc='best')

        plt.tight_layout()

    return frame_to_rows


def collect_deviating_rows_for_channels(
        frames: np.ndarray,
        active_channels: list,
        threshold_factor: int = 3
) -> dict:
    """
    Collect all deviating rows across channels for each frame.

    Parameters
    ----------
    frames : np.ndarray
        Imaging data, shape (n_frames, n_channels, height, width).
    active_channels : list
        List of channel indices to process.
    threshold_factor : int, optional
        Threshold factor for deviation detection.

    Returns
    -------
    dict
        Mapping from frame index to sorted list of deviating rows.
    """
    channel_deviations = {}
    for channel_idx in active_channels:
        channel_deviations[channel_idx] = identify_rows_deviation(
            frames[:, channel_idx, :, :],
            threshold_factor=threshold_factor,
            plot=False)

    deviating_rows = {}
    all_frames = set()
    for channel_deviation in channel_deviations.values():
        all_frames.update(channel_deviation.keys())

    for frame in all_frames:
        rows_for_frame = []
        for channel_deviation in channel_deviations.values():
            rows_for_frame.extend(channel_deviation.get(frame, []))
        deviating_rows[frame] = sorted(set(rows_for_frame))

    return dict(sorted(deviating_rows.items()))