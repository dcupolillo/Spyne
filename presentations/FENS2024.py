""" Created on Mon Jun 10 10:25:51 2024
    @author: dcupolillo """

import numpy as np
import spyne
from spyne.display.plot_segmenter import collect_centroid_fov
import matplotlib.pyplot as plt
from pathlib import Path
from neuronpath.path import neuronpath
from typing import Callable, Dict, List, Tuple
import ROIpy as rp
from collections import Counter
from matplotlib.colors import LinearSegmentedColormap
from matplotlib import cm
import scipy.stats as stats
from matplotlib.animation import FuncAnimation, FFMpegWriter

paths = neuronpath('240201', 1)
dataset = spyne.ImagingDataset(paths.imaging)
segmenter = spyne.DatasetSegmenter(dataset)

# zscores_data = fl.load(Path(paths, 'batch_zscore_data.h5'))
# dff_data = fl.load(Path(paths, 'batch_dFF_data.h5'))
# timestamps_data = fl.load(Path(paths, 'batch_ts_data.h5'))
# spines_data = fl.load(Path(paths, 'batch_spines_data.h5'))


def extract_traces_as_np(
        data: dict
) -> np.ndarray:
    """
    Extract all traces from the given data dictionary
    and return them as a 2D numpy array.

    Parameters
    ----------
    data : dict
        Dictionary containing the trace data.

    Returns
    -------
    np.ndarray
        A 2D numpy array where each row corresponds to a trace
        from the input data.
    """

    all_traces = []

    for (roi_idx, spine_idx), spines_data in data.items():
        for spine_n, traces in spines_data.items():
            for trace in traces:
                all_traces.append(trace)

    return np.array(all_traces)


def extract_mean_traces(
        data: dict
) -> dict:
    """
    Extract the mean trace for each spine from the given data dictionary.

    Parameters
    ----------
    data : dict
        Dictionary containing the trace data.

    Returns
    -------
    dict
        A dictionary with the same structure as the input data,
        but with each value being the mean trace for the corresponding spine.
    """

    mean_traces = {}

    for (roi_idx, spine_idx), spines_data in data.items():
        mean_traces[(roi_idx, spine_idx)] = {}
        for spine_n, traces in spines_data.items():
            traces_array = np.array(traces)
            mean_trace = np.mean(traces_array, axis=0)
            mean_traces[(roi_idx, spine_idx)][spine_n] = mean_trace

    return mean_traces


def normalize_traces(
        data: Dict[Tuple[int, int], Dict[int, np.ndarray]]
) -> Dict[Tuple[int, int], Dict[int, np.ndarray]]:
    """
    Normalize all traces in the given data dictionary to a scale of 0-1,
    where 1 is the maximum value of the trace and 0 is the minimum value.

    Parameters
    ----------
    data : dict
        Dictionary containing the trace data.

    Returns
    -------
    dict
        A dictionary with the same structure as the input data,
        but with each value being the normalized trace
        for the corresponding spine.
    """

    normalized_traces = {}

    for (roi_idx, spine_idx), spines_data in data.items():
        normalized_traces[(roi_idx, spine_idx)] = {}
        for spine_n, trace in spines_data.items():
            trace_min = np.min(trace)
            trace_max = np.max(trace)
            normalized_trace = (trace - trace_min) / (trace_max - trace_min)
            normalized_traces[(roi_idx, spine_idx)][spine_n] = normalized_trace

    return normalized_traces


def get_frame_index_from_t(
        timestamp_data: dict,
        segmenter: object,
        t: float
) -> dict:
    """
   Calculate the frame indices for t second for each ROI and spine.

   Parameters
   ----------
   timestamps_data : dict
       Dictionary containing timestamp traces for each ROI and spine.
   segmenter : object
       The DatasetSegmenter object containing the dataset and ROIs.
   t : float
       The time of the frame index to extract.

   Returns
   -------
   frame_indices : dict
       Dictionary with the frame indices for t=1.00 second
       for each ROI and spine.
   """

    frame_indices = {}

    for key, roi_timestamps in timestamp_data.items():
        frame_indices[key] = {}

        for spine_key, timestamps in roi_timestamps.items():
            timestamps = np.array(timestamps)
            closest_frame_index = np.argmin(np.abs(timestamps - t))
            frame_indices[key][spine_key] = closest_frame_index

    return frame_indices


def align_pad_and_sort_traces(
        data: dict,
        frame_indices: Dict[Tuple[int, int], Dict[int, int]],
        sorting_criterion: Callable[[np.ndarray], float]
) -> Tuple[np.ndarray, List[float], List[int]]:
    """
    Align traces to a specific frame, pad them with NaN values,
    and sort them based on a provided criterion.

    Parameters
    ----------
    data : dict
        Dictionary containing the trace data.
    frame_indices : Dict[Tuple[int, int], Dict[int, int]]
        Dictionary containing the frame indices to align the traces.
    sorting_criterion : Callable[[np.ndarray], float]
        Function to calculate the sorting value for each trace.
        The function takes a trace (numpy array) as input
        and returns a float value used for sorting.

    Returns
    -------
    Tuple[np.ndarray, List[float], List[int]]
        A tuple containing:
        - A 2D numpy array where each row corresponds
          to an aligned, padded, and sorted trace.
        - A list of sorting values used to sort the traces.
        - A list of sorted indices.
    """
    aligned_traces = []
    sorting_values = []

    # Find the maximum frame index for alignment
    max_frame_index = max(
        frame_index
        for roi_traces in frame_indices.values()
        for frame_index in roi_traces.values()
    )

    for key, roi_traces in data.items():
        for spine_key, traces in roi_traces.items():
            frame_index_at_1s = int(frame_indices[key][spine_key])
            padding_length = max_frame_index - frame_index_at_1s

            # Check if the entry is a single trace (1D array) or multiple traces (2D array)
            if traces.ndim == 1:
                traces = [traces]
            
            for trace in traces:
                # Pad the trace so that the "1 sec frame" aligns at the same index
                padded_trace = np.pad(
                    trace, (padding_length, 0), constant_values=np.nan
                )
                aligned_traces.append(padded_trace)

                # Calculate the sorting value based on the first 5 frames
                sorting_value = sorting_criterion(
                    padded_trace[max_frame_index:max_frame_index + 5])
                sorting_values.append(sorting_value)

    # Ensure all traces have the same length by padding the end with NaNs
    max_length = max(len(trace) for trace in aligned_traces)
    aligned_traces_padded = [
        np.pad(trace, (0, max_length - len(trace)), constant_values=np.nan)
        for trace in aligned_traces
    ]

    aligned_traces_array = np.vstack(aligned_traces_padded)

    sorted_indices = np.argsort(sorting_values)[::-1]
    sorted_aligned_traces = aligned_traces_array[sorted_indices]
    sorted_values = [sorting_values[i] for i in sorted_indices]

    return sorted_aligned_traces, sorted_values, sorted_indices


def sorting_criterion(trace: np.ndarray) -> float:
    return np.nanmean(trace[:5])


# %% Mean activity in spines heatmap

zscores = extract_traces_as_np(segmenter.batch_z_scores)
mean_zscores_data = extract_mean_traces(segmenter.batch_z_scores)

frame_indices_at_1s = get_frame_index_from_t(
    segmenter.batch_ts, segmenter, 1.00)

(
 sorted_aligned_traces,
 sorted_values,
 sorted_indices
 ) = align_pad_and_sort_traces(
    mean_zscores_data, frame_indices_at_1s, sorting_criterion
)

max_frame_index = max(
    frame_index
    for roi_traces in frame_indices_at_1s.values()
    for frame_index in roi_traces.values()
)


# Plot the heatmap
fontsize = 28
fig, ax = plt.subplots(figsize=(6, 12))
heatmap = ax.imshow(
    sorted_aligned_traces,
    aspect='auto',
    cmap='viridis',
    interpolation='none')

# Customize the color bar
cbar = plt.colorbar(heatmap, ax=ax, pad=0.05, label='Z-Score')
cbar.outline.set_edgecolor('black')
cbar.outline.set_linewidth(0.5)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_label('Z-score', fontsize=fontsize, labelpad=0.02)

# Set the axis labels with increased font size
ax.set_xlabel('Frame from stim.', fontsize=fontsize)
ax.set_ylabel('Spines', fontsize=fontsize)

# Set y-ticks with increased font size
num_traces = sorted_aligned_traces.shape[0]
yticks = np.arange(0, num_traces, 500)
ax.set_yticks(yticks)
ax.set_yticklabels(yticks, fontsize=fontsize)

num_frames = sorted_aligned_traces.shape[1]
spacing = int(max_frame_index / 2)

xticks = np.arange(0, num_frames, spacing)
x_labels = [str(x - max_frame_index) for x in xticks]
ax.set_xticks(xticks)
ax.set_xticklabels(x_labels, fontsize=fontsize)
ax.axvline(x=max_frame_index,
           ymin=1,
           ymax=1.01,
           color='black',
           lw=2,
           clip_on=False)
ax.set_xlim(5.5, 30.5)

# Set x-tick labels with increased font size
ax.tick_params(axis='x', labelsize=fontsize)
ax.tick_params(axis='y', labelsize=fontsize)

plt.tight_layout()
plt.subplots_adjust(right=0.8)

# plt.savefig(
#     Path(r'Y:\dcupolillo\Conferences and meetings\Gordon2024'
#           r'\zscores_heatmap_240201_cell0001.png'),
#     dpi=1000)

# %%

zscore = segmenter.batch_z_scores[30, 4][7][0]
ts = segmenter.batch_ts[30, 4][7][0]

fig, ax = plt.subplots(figsize=(10, 4))
fontsize = 24
ax.plot(ts, zscore, color='black', clip_on=False)
std = np.std(zscore)
ax.axhline(0, color='gray', ls='--', alpha=0.5, clip_on=False)
ax.axhline(1.96 * std, color='red', alpha=0.5)

ax.spines[['right', 'top']].set_visible(False)
ax.spines[['bottom', 'left']].set_position(('outward', 20))

ax.text(x=4, y=(1.96 * std), s='1.96 x S.D.', va='center', fontsize=fontsize)
ax.set_ylim(-5, 5)
ax.set_xlim(0, 4)
ax.set_xlabel('Time (s)', fontsize=fontsize)
ax.set_ylabel('Z-score', fontsize=fontsize)
ax.tick_params(labelsize=fontsize)

plt.tight_layout()

# plt.savefig(
#     Path(r'Y:\SynEmo\figures4poster\example_trace_with_1.96_SD.png'),
#     dpi=1000)

# %%

morph = rp.Morphology(paths.stackpath, paths.tracepath)
sf = rp.Scanfields(paths.stackpath, paths.tracepath)

fig, ax = plt.subplots(figsize=(8, 8))

morph.plot(morph.neuron, ax=ax)
sf.plot(sf.neuComp[30], ax=ax, edgecolor='#0072AF', linewidth=3)

ax.plot([-200, -100], [-150, -150], color='black', linewidth=3)

ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])

ax.set_aspect('equal')

# plt.savefig(
#     Path(r'Y:\SynEmo\figures4poster\skeleton_Z=30_240123_cell0001.png'),
#     dpi=1000)

# %%


def dFF(
        frame_seq: np.ndarray,
        frame_rate: float,
        rolling_bsl: str = 'centered',
        apply_filter: str = 'okada'
) -> np.ndarray:

    n_frames, _, height, width = frame_seq.shape

    f = np.zeros((n_frames, height, width))

    for n in range(n_frames):
        frame = frame_seq[n, 0, :, :]
        frame_min = np.min(frame)
        frame_rescaled = frame - frame_min
        frame_rescaled = frame_rescaled.astype(np.float32)
        f[n] = frame_rescaled

    # Centered rolling quantile
    window_sec = 1
    window = int(window_sec * frame_rate)
    half_window = int(window / 2)
    min_quantile = 10

    bl = np.zeros_like(f)

    for i in range(height):
        for j in range(width):
            pixel_f = f[:, i, j]
            if rolling_bsl == 'centered':
                baseline = [np.percentile(
                    pixel_f[
                        max(0, t - half_window): min(
                            n_frames, t + half_window + 1)], min_quantile)
                            for t in range(n_frames)]
            else:
                baseline = [np.percentile(pixel_f[t: t + window], min_quantile)
                            for t in range(n_frames - window + 1)]
                baseline = np.concatenate(
                    [baseline, np.repeat(baseline[-1], window - 1)])

            bl[:, i, j] = baseline

    dff = (f - bl) / bl

    dff = apply_okada_filter(dff)

    return dff


def apply_okada_filter(dff: np.ndarray) -> np.ndarray:
    """
    Apply Okada filter to each pixel in the dFF array.

    Parameters
    ----------
    dff : np.ndarray
        The dFF array of shape (n_frames, height, width).

    Returns
    -------
    np.ndarray
        The filtered dFF array of shape (n_frames, height, width).
    """
    filtered_series = np.copy(dff)
    n_frames, height, width = dff.shape

    for i in range(height):
        for j in range(width):
            pixel_ts = dff[:, i, j]
            mean = np.mean(pixel_ts)
            st_dev = np.std(pixel_ts)

            for t in range(1, n_frames - 1):
                xt = pixel_ts[t]
                xt_minus_1 = pixel_ts[t - 1]
                xt_plus_1 = pixel_ts[t + 1]

                # Z is defined as signal saliency against background noise
                Z = np.abs((xt_plus_1 - mean) / st_dev)

                # Check if xt is the median
                if (xt - xt_minus_1) * (xt - xt_plus_1) > 0:
                    filtered_series[t, i, j] = (
                        (xt_minus_1 + Z * xt + xt_plus_1) / (2 + Z))

    return filtered_series


dff_values = dFF(dataset[30][4][0].ch1.frame_seq, frame_rate=14.8122)

normalized_dff_values = (
    dff_values - dff_values.min(axis=(1, 2), keepdims=True)) / (
        dff_values.max(axis=(1, 2), keepdims=True) -
        dff_values.min(axis=(1, 2), keepdims=True))

frame_index = 28

plt.figure(figsize=(10, 8))
fontsize = 42
heatmap = plt.imshow(
    dff_values[frame_index],
    cmap='hot',
    interpolation='nearest',)
colorbar = plt.colorbar(heatmap)
colorbar.set_label(label=r'$\Delta F/F_0$', labelpad=15, fontsize=fontsize)
colorbar.ax.tick_params(labelsize=fontsize)
plt.xticks([])
plt.yticks([])

# plt.savefig(
#     Path(r'Y:\dcupolillo\Conferences and meetings\Gordon2024'
#          fr'\frame{frame_index}_dFF_heatmap_240123[30][4].png'),
#     dpi=1000)

# %%


# ax.plot([-200, -100], [-150, -150], color='black', linewidth=3)


def plot_traces_with_high_points(
    data: Dict[Tuple[int, int], Dict[int, np.ndarray]],
    increment: int,
    consecutive_points: int,
    std_factor: float,
    start_edge: int,
    end_edge: int
) -> None:
    """
    Plot traces and highlight segments where a number of
    consecutive points exceed a threshold.

    Parameters
    ----------
    data : Dict[Tuple[int, int], Dict[int, np.ndarray]]
        Dictionary containing the trace data.
    increment : int
        The value to increment the offset for each plot.
    consecutive_points : int
        The number of consecutive points that must exceed the threshold.
    std_factor : float
        The factor by which to multiply the standard deviation
        for the threshold.
    start_edge : int
        The starting index of the edge where to look for points
        exceeding the threshold.
    end_edge : int
        The ending index of the edge where to look for points
        exceeding the threshold.
    """
    offset = 0

    traces_with_event = []

    for key, roi_traces in data.items():
        for spine_key, traces in roi_traces.items():
            for trace_n, trace in enumerate(traces):
                std = np.std(trace)
                plotted = False  # Flag to track if the trace has been plotted
                for n in range(start_edge, end_edge - consecutive_points + 1):
                    if all(trace[n + i] > (std_factor * std)
                           for i in range(consecutive_points)):
                        if not plotted:
                            plt.plot(trace + offset, color='black')
                            plotted = True
                        plt.plot(range(n, n + consecutive_points),
                                 trace[n:n + consecutive_points] + offset,
                                 color='red')
                        traces_with_event.append([key, spine_key, trace_n])
                        offset += increment
                        break  # Exit the loop after plotting the trace once

    return traces_with_event


traces_with_event = plot_traces_with_high_points(
    data=segmenter.batch_z_scores,
    increment=5,
    consecutive_points=2,
    std_factor=1.96,
    start_edge=16,
    end_edge=21
)

all_roi = [z[0] for z in traces_with_event]
all_roi_spines = [(z[0], z[1]) for z in traces_with_event]
spine_counter = Counter((trace[0], trace[1]) for trace in traces_with_event)

# spines_data_restructured = {}

# for roi in segmenter.batch_spines_data:
#     for spine_dict in roi:
#         scanfield = spine_dict['scanfield']
#         roi_index = spine_dict['roi_index']
#         spine_id = spine_dict['spine_id']
#         centroid_pix = spine_dict['centroid_pix']
#         centroid_fov = spine_dict['centroid_fov']
#         mask = spine_dict['mask']

#         if (scanfield, roi_index) not in spines_data_restructured:
#             spines_data_restructured[(scanfield, roi_index)] = {}

#         if spine_id not in spines_data_restructured[(scanfield, roi_index)]:
#             spines_data_restructured[(scanfield, roi_index)][spine_id] = {}

#         spines_data_restructured[
#             (scanfield, roi_index)][spine_id]['centroid_pix'] = centroid_pix
#         spines_data_restructured[
#             (scanfield, roi_index)][spine_id]['centroid_fov'] = centroid_fov
#         spines_data_restructured[
#             (scanfield, roi_index)][spine_id]['mask'] = mask


centroids_with_event = []
counts = []

# TODO: integrate z?
for roi in segmenter.batch_spines_data:
    for key, spines in roi.items():

        if key in [trace[0] for trace in traces_with_event]:

            for n_spines, spine in enumerate(spines):

                if (key, n_spines) in [
                        (trace[0], trace[1]) for trace in traces_with_event]:

                    centroid = (
                        spine['centroid_fov'] +
                        [spine['roi_z']])
                    centroids_with_event.append(centroid)
                    counts.append(spine_counter[(key, n_spines)])

centroids_with_event = np.array(centroids_with_event)


# Normalize the counts for color mapping
max_count = max(counts)
norm = plt.Normalize(vmin=0, vmax=5)
cmap = cm.inferno
colors = cmap(norm(counts))


# Plot the centroids with color coding based on counts
fig, ax = plt.subplots(figsize=(16, 16))
fontsize = 32
ax.set_aspect('equal')

segmenter.plot_all_spines(
    ax=ax,
    spine_color='gray',
    spine_size=10)
scatter = ax.scatter(
    centroids_with_event[:, 0],
    centroids_with_event[:, 1],
    c=colors,
    s=20)

cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax)
cbar.set_label('Frequency', fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_ticks(np.arange(6))
cbar.set_ticklabels([f'{i}' for i in range(6)])

ax.spines[['right', 'top', 'bottom', 'left']].set_visible(False)
ax.set_xticks([])
ax.set_yticks([])
ax.set_ylabel('')
ax.set_xlabel('')

# plt.savefig(
#     Path(r'Y:\dcupolillo\Conferences and meetings\Gordon2024'
#           r'\240201_cell0001_mapped_spines_with events.png'),
#     dpi=1000)


# %% plot spines in 3d and animate


# Plot the centroids with color coding based on counts in 3D
fig = plt.figure(figsize=(16, 16))
ax = fig.add_subplot(111, projection='3d')
fontsize = 32

# Plot all spines
all_spines = segmenter.batch_spines_data
all_centroids = np.array(collect_centroid_fov(all_spines))

# Apply objective resolution to z coordinates of all centroids
all_centroids[:, 2] = all_centroids[:, 2] / morph.objective_resolution
# centroids_with_event[:, 2] = centroids_with_event[:, 2] / morph.objective_resolution

# Find the unique points in all_centroids that are not in centroids_with_event
all_centroids_set = set(map(tuple, all_centroids))
centroids_with_event_set = set(map(tuple, centroids_with_event))
unique_all_centroids = np.array(list(all_centroids_set - centroids_with_event_set))

if len(all_centroids) == 0:
    raise ValueError("No spine data found in the segmenter.")

# Plot unique gray points
ax.scatter(unique_all_centroids[:, 0],
           unique_all_centroids[:, 1],
           unique_all_centroids[:, 2],
           s=10,
           color='gray',
           edgecolors='none',
           alpha=0.4,
           zorder=-1,
           depthshade=True)

# Scatter plot for centroids with events
scatter = ax.scatter(
    centroids_with_event[:, 0],
    centroids_with_event[:, 1],
    centroids_with_event[:, 2],
    c=colors,
    s=30,
    zorder=25,
    depthshade=False)


# Colorbar
cbar = plt.colorbar(plt.cm.ScalarMappable(cmap=cmap, norm=norm), ax=ax, shrink=0.5, aspect=5)
cbar.set_label('Frequency', fontsize=fontsize)
cbar.ax.tick_params(labelsize=fontsize)
cbar.set_ticks(np.arange(6))
cbar.set_ticklabels([f'{i}' for i in range(6)])

ax.set_xticks([])
ax.set_yticks([])
ax.set_zticks([])

ax.view_init(elev=30, azim=45)
ax.set_aspect('equal')

# Function to update the view angle
def update_view(num):
    ax.view_init(elev=30, azim=num)

# Create the animation
ani = FuncAnimation(fig, update_view, frames=np.arange(0, 360, 1), interval=50)
ani.save(
    Path(r'Y:\dcupolillo\Conferences and meetings\Gordon2024'
          r'\240201_cell0001_animated_spines_plot.mp4'), writer='ffmpeg')

# %%

occurrence_counts = Counter(spine_counter.values())

# Data for the plots
labels = [f"{i}" for i in range(1, 6)]
sizes = [occurrence_counts.get(i, 0) for i in range(1, 6)]
colors_list = [(1, 0, 0), (0, 0, 0)]  # Black to red
cumulative_counts = [sum(sizes[:i]) for i in range(1, len(sizes) + 1)]
cmap = LinearSegmentedColormap.from_list("black_red", colors_list, N=256)
colors = cmap(norm(sizes))

# Data for the stacked bar plot
total_spines = len(spine_counter)  # Total number of unique spines
labels = [f"{i}" for i in range(1, 6)]
counts = [occurrence_counts.get(i, 0) for i in range(1, 6)]
fractions = [count / total_spines for count in counts]

# Prepare data for stacked bar plot
categories = ['1', '2', '3', '4', '5']
width = 0.5

fig, ax = plt.subplots(figsize=(12, 3))
bottom = 0

ax.spines[['right', 'top', 'left']].set_visible(False)
ax.spines[['bottom', 'left']].set_position(('outward', 20))

for fraction, label, color in zip(fractions, categories, colors[::-1]):
    ax.barh([0], [fraction],
            height=width,
            left=bottom,
            label=label,
            color=color)
    bottom += fraction

ax.set_xlim(0, 1)
ax.set_ylim(-0.75, 0.75)
ax.set_yticks([0])
ax.set_yticklabels([''],
                   va='center',
                   ha='right',
                   fontsize=fontsize)
ax.set_xticks(np.linspace(0, 1, 6))
ax.set_xticklabels([f'{tick:.1f}' for tick in np.linspace(0, 1, 6)],
                   fontsize=fontsize)
ax.set_xlabel('Fraction', fontsize=fontsize)

plt.tight_layout()
plt.show()

# plt.savefig(
#     Path(r'Y:\SynEmo\figures4poster\events_per_spine_frequency.png'),
#     dpi=1000)


# %%


def mean_and_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    ci = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, ci


def sholl_analysis(soma: Tuple[float, float],
                   spines: List[Tuple[float, float]],
                   start_radius: float,
                   radius_increment: float,
                   num_radii: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform Sholl analysis on given spines coordinates around a central soma.

    Parameters
    ----------
    soma : Tuple[float, float]
        The (x, y) coordinates of the soma.
    spines : List[Tuple[float, float]]
        List of (x, y) coordinates of the spines.
    start_radius : float
        The starting radius for the concentric circles.
    radius_increment : float
        The increment value for the radii.
    num_radii : int
        The number of radii to be considered.

    Returns
    -------
    radii : np.ndarray
        Array of radii used in the analysis.
    intersections : np.ndarray
        Array of intersection counts corresponding to each radius.
    """
    radii = start_radius + np.arange(num_radii) * radius_increment
    intersections = np.zeros(num_radii)

    for i, radius in enumerate(radii):
        inner_radius = radius - radius_increment
        for spine in spines:
            distance = np.sqrt(
                (spine[0] - soma[0])**2 + (spine[1] - soma[1])**2)
            if inner_radius < distance <= radius:
                intersections[i] += 1

    return radii, intersections


def cumulative_sholl_analysis(soma: Tuple[float, float],
                              spines: List[Tuple[float, float]],
                              start_radius: float,
                              radius_increment: float,
                              num_radii: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Perform cumulative Sholl analysis on given spines
    coordinates around a central soma.

    Parameters
    ----------
    soma : Tuple[float, float]
        The (x, y) coordinates of the soma.
    spines : List[Tuple[float, float]]
        List of (x, y) coordinates of the spines.
    start_radius : float
        The starting radius for the concentric circles.
    radius_increment : float
        The increment value for the radii.
    num_radii : int
        The number of radii to be considered.

    Returns
    -------
    radii : np.ndarray
        Array of radii used in the analysis.
    cumulative_intersections : np.ndarray
        Array of cumulative intersection counts corresponding to each radius.
    """
    radii = start_radius + np.arange(num_radii) * radius_increment
    cumulative_intersections = np.zeros(num_radii)

    for i, radius in enumerate(radii):
        for spine in spines:
            distance = np.sqrt(
                (spine[0] - soma[0])**2 + (spine[1] - soma[1])**2)
            if distance <= radius:
                cumulative_intersections[i] += 1

    return radii, cumulative_intersections


start_radius = 20  # Starting radius
radius_increment = 10  # Radius increment
num_radii = 25  # Number of radii

neurons_to_analyze = [
    ('240123', 1),
    ('240130', 1),
    ('240201', 1),
]

fig, ax = plt.subplots(figsize=(8, 6))
fontsize = 32
ax.spines[['right', 'top']].set_visible(False)
ax.spines[['bottom', 'left']].set_position(('outward', 30))
for spine in ax.spines.values():
    spine.set_linewidth(2)
ax.set_xlabel('Distance from soma (µm)', fontsize=fontsize)
ax.set_ylabel('Active spine number', fontsize=fontsize)
ax.tick_params(
    axis='both',
    which='major',
    width=2,
    length=10,
    labelsize=fontsize)
ax.set_ylim(0, 30)
ax.set_xlim(0, 300)
plt.tight_layout()

fig_cumulative, ax_cumulative = plt.subplots(figsize=(8, 6))
ax_cumulative.spines[['right', 'top']].set_visible(False)
ax_cumulative.spines[['bottom', 'left']].set_position(('outward', 30))
for spine in ax_cumulative.spines.values():
    spine.set_linewidth(2)
ax_cumulative.set_xlabel('Distance from soma (µm)', fontsize=fontsize)
ax_cumulative.set_ylabel(
    'Cumulative\nactive spine distribution', fontsize=fontsize)
ax_cumulative.tick_params(
    axis='both',
    which='major',
    width=2,
    length=10,
    labelsize=fontsize)
ax_cumulative.set_ylim(0, 200)
ax_cumulative.set_xlim(0, 300)
plt.tight_layout()

intersection_all = np.array([None] * len(neurons_to_analyze))
intersection_cumulative_all = [None] * len(neurons_to_analyze)

for n, (path, cell_n) in enumerate(neurons_to_analyze):
    paths = neuronpath(path, cell_n)
    assert paths
    dataset = spyne.ImagingDataset(paths.imaging)
    segmenter = spyne.DatasetSegmenter(dataset)

    morph = rp.Morphology(paths.stackpath, paths.tracepath)
    soma = (morph.soma.x_deg, morph.soma.y_deg)

    spines_data_restructured = {}

    for roi in segmenter.batch_spines_data:
        for spine_dict in roi:
            scanfield = spine_dict['scanfield']
            roi_index = spine_dict['roi_index']
            spine_id = spine_dict['spine_id']
            centroid_pix = spine_dict['centroid_pix']
            centroid_fov = spine_dict['centroid_fov']
            mask = spine_dict['mask']

            if (scanfield, roi_index) not in spines_data_restructured:
                spines_data_restructured[(scanfield, roi_index)] = {}

            if spine_id not in spines_data_restructured[(scanfield, roi_index)]:
                spines_data_restructured[(scanfield, roi_index)][spine_id] = {}

            spines_data_restructured[
                (scanfield, roi_index)][spine_id]['centroid_pix'] = centroid_pix
            spines_data_restructured[
                (scanfield, roi_index)][spine_id]['centroid_fov'] = centroid_fov
            spines_data_restructured[
                (scanfield, roi_index)][spine_id]['mask'] = mask

    centroids_with_event = []
    for key, roi in spines_data_restructured.items():
        if key in [trace[0] for trace in traces_with_event]:
            for key_spine, spine in roi.items():
                if (key, key_spine) in [
                        (trace[0], trace[1]) for trace in traces_with_event]:
                    centroids_with_event.append(spine['centroid_fov'])
    centroids_with_event = np.array(centroids_with_event)

    centroids_with_event_um = [[c[0]*morph.objective_resolution,
                                c[1]*morph.objective_resolution]
                               for c in centroids_with_event]

    radii, intersections = sholl_analysis(
        soma,
        centroids_with_event_um,
        start_radius,
        radius_increment,
        num_radii)
    intersection_all[n] = intersections

    radii_cumulative, intersections_cumulative = cumulative_sholl_analysis(
        soma,
        centroids_with_event_um,
        start_radius,
        radius_increment,
        num_radii)
    intersection_cumulative_all[n] = intersections_cumulative

    ax.plot(
        radii,
        intersections,
        color='#B95B46',
        clip_on=False,
        lw=0.8,
        alpha=0.5,)

    ax_cumulative.plot(
        radii_cumulative,
        intersections_cumulative,
        clip_on=False,
        color='#B95B46',
        lw=0.8,
        alpha=0.5,)

    print(f'{path}_cell{cell_n}: activa spines: {sum(intersections)}')


CI_low = (np.mean(intersection_all, axis=0) - 1.96
          * stats.sem(intersection_all, axis=0))
CI_high = (np.mean(intersection_all) + 1.96
           * stats.sem(intersection_all))

CI_low_cumulative = (np.mean(intersection_cumulative_all, axis=0) - 1.96
                     * stats.sem(intersection_cumulative_all, axis=0))
CI_high_cumulative = (np.mean(intersection_cumulative_all, axis=0) + 1.96
                      * stats.sem(intersection_cumulative_all, axis=0))


ax.plot(
    radii,
    np.mean(intersection_all, axis=0),
    color='black',
    linewidth=2)
ax.fill_between(
    radii,
    CI_low,
    CI_high,
    color='#46A4B9',
    alpha=0.5,
    clip_on=False,
    linewidth=0,)

ax_cumulative.plot(
    radii_cumulative,
    np.mean(intersection_cumulative_all, axis=0),
    color='black',
    linewidth=2)
ax_cumulative.fill_between(
    radii_cumulative,
    CI_low_cumulative,
    CI_high_cumulative,
    color='#46A4B9',
    alpha=0.5,
    clip_on=False,
    linewidth=0,)

# plt.savefig(
#     Path(r'Y:\dcupolillo\Conferences and meetings\Gordon2024'
#           r'\active_spine_number_from_soma.png'),
#     dpi=1000)


# plt.savefig(
#     Path(r'Y:\dcupolillo\Conferences and meetings\Gordon2024'
#           r'\active_spine_cumulative_from_soma.png'),
#     dpi=1000)

# %%



zscores_of_active_spines = []
zscores_of_inactive_spines = []

traces_with_event_set = set(
    (trace[0], trace[1], trace[2]) for trace in traces_with_event)

for key, roi in segmenter.batch_z_scores.items():
    for key_spine, spine in roi.items():
        for sweep_n, sweep in enumerate(spine):
            if (key, key_spine, sweep_n) in traces_with_event_set:
                zscores_of_active_spines.append(sweep)
            else:
                zscores_of_inactive_spines.append(sweep)

mean_active, ci_active = mean_and_ci(zscores_of_active_spines)
mean_inactive, ci_inactive = mean_and_ci(zscores_of_inactive_spines)


fig, ax = plt.subplots()
ax.plot(mean_active, color='blue')
ax.fill_between(
    range(len(mean_active)),
    mean_active - ci_active,
    mean_active + ci_active,
    color='blue',
    alpha=0.3)
