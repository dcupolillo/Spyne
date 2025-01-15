""" Created on Wed Sep 18 13:27:51 2024
    @author: dcupolillo """

import ROIpy as rp
import spynalyzer as spa
from neuronpath.path import neuronpath
from dataplotter import DataPlotter
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from scipy import stats
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.gridspec as gridspec
from matplotlib.gridspec import GridSpec
from spynalyzer.utils import node_spine_distance
import actionpytential as ap
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from matplotlib.colors import Normalize


# %% Load data

neuronpaths_list = [
    neuronpath('240828', 1),
    neuronpath('240910', 1),
    neuronpath('240912', 2),
    neuronpath('240913', 1),
    neuronpath('240813', 1),
    neuronpath('240814', 2),
    neuronpath('240827', 2),
]

spynalyzer_list = [
    spa.SpyneAnalyzer(path) for path in neuronpaths_list]

traces_list = [
    ap.EphyDataset(path) for path in neuronpaths_list]

morphology_list, scanfields_list, dataset_list, segmenter_list = zip(*[
    (element.morph, element.sf, element.dataset, element.segmenter)
    for element in spynalyzer_list
])


# %% Plotting parameters

saving_path = Path(r'Y:\SynEMO\SfN2024')

fontsize = 18
single_box_width = 0.2
single_group_strip_offset = 0.3
single_group_meanline_width = 0.1
single_group_figsize = (2.5, 3)  # width, height

BLA_cmap = LinearSegmentedColormap.from_list("cmap1", ['#46A4B9', 'black'])
CA3_cmap = LinearSegmentedColormap.from_list("cmap2", ['#B95B46', 'black'])

# %% Plot overall spines map with color-coded number of events

fontsize = 30
for n, segmenter in enumerate(segmenter_list):
    fig, axes = plt.subplots(1, 2, sharex=True, sharey=True, figsize=(14, 12))

    for ax in axes:
        segmenter.plot_all_spines(spine_color='#bec0c2', ax=ax)

    segmenter.plot_events_spines(
        'BLA', cmap=BLA_cmap, ax=axes[0], fontsize=fontsize)
    segmenter.plot_events_spines(
        'CA3', cmap=CA3_cmap, ax=axes[1], fontsize=fontsize)

    for ax in axes:
        ax.set_aspect('equal')
        for spine in ['top', 'bottom', 'left', 'right']:
            ax.spines[spine].set_visible(False)

        ax.set_ylabel('', fontsize=fontsize)
        ax.set_xlabel('', fontsize=fontsize)
        ax.tick_params(left=False, bottom=False, right=False, top=False)
        ax.set_xticklabels([])
        ax.set_yticklabels([])

    plt.tight_layout()

    # if n == 2:
    #     fig.savefig(
    #         saving_path / 'spines_map.png',
    #         dpi=600)


# %% Plot event spines map with increasing event number threshold

fontsize = 18

for n, segmenter in enumerate(segmenter_list):

    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(15, 8))

    for row, region, cmap in zip(
            [0, 1], ['BLA', 'CA3'], [BLA_cmap, CA3_cmap]):

        for n_event_threshold in range(5):

            segmenter.plot_all_spines(
                spine_color='#bec0c2',
                ax=axes[row][n_event_threshold])

            try:
                segmenter.plot_events_spines(
                    region,
                    n_event_threshold=n_event_threshold,
                    cmap=cmap,
                    show_cmap=False,
                    ax=axes[row][n_event_threshold],
                    fontsize=fontsize
                )
            except ValueError:
                pass

            axes[0][n_event_threshold].set_title(
                f"n events:\n{n_event_threshold + 1} or more"
                if n_event_threshold < 4
                else f"n events:\n{n_event_threshold + 1}",
                fontsize=fontsize)

    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')
            ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
            ax.set_ylabel('')
            ax.set_xlabel('')
            ax.tick_params(left=False, bottom=False, right=False, top=False)
            ax.set_xticklabels([])
            ax.set_yticklabels([])

    axes[0][0].set_ylabel('BLA', fontsize=fontsize)
    axes[1][0].set_ylabel('CA3', fontsize=fontsize)

    plt.tight_layout()

    # if n == 2:
    #     fig.savefig(
    #         saving_path / 'spines_map_increasing_events.png',
    #         dpi=600)


# %% Percentage of spines with at least 1 event

BLA_event_percentage = np.empty(len(segmenter_list))
CA3_event_percentage = np.empty(len(segmenter_list))

for n, segmenter in enumerate(segmenter_list):

    spines_with_BLA_event = sum(
        [1 if sum(flags) > 0 else 0
         for flags in segmenter.calcium_events_BLA])
    spines_with_CA3_event = sum(
        [1 if sum(flags) > 0 else 0
         for flags in segmenter.calcium_events_CA3])

    all_spines = len(segmenter.batch_spines_data)

    BLA_event_percentage[n] = spines_with_BLA_event / all_spines * 100
    CA3_event_percentage[n] = spines_with_CA3_event / all_spines * 100

spines_percentages_dataframe = pd.DataFrame({
    "Input type": (["BLA"] * len(BLA_event_percentage) +
                   ["CA3"] * len(CA3_event_percentage)),
    "Spines percentage": np.concatenate((
        BLA_event_percentage, CA3_event_percentage))
})

spines_percentages = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(15, 25),
    x_lims=(0, 1),
    y_label='Spines with event %',
    x_ticks=([0, 1]),
    x_tick_labels=(['BLA', 'CA3']),
    palette=['#46A4B9', '#B95B46'],
    y_pad=18,
    )

spines_percentages.box_plot(
    spines_percentages_dataframe,
    x='Input type',
    y='Spines percentage',
    box_width=0.15,
    hue='Input type',)

spines_percentages.strip_plot(
    spines_percentages_dataframe,
    x='Input type',
    y='Spines percentage',
    linewidth=1.5,
    edgecolor='#808080',
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

spines_percentages.display_mean(
    spines_percentages_dataframe,
    x='Input type',
    y='Spines percentage',
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

bla_points = spines_percentages_dataframe[
    spines_percentages_dataframe['Input type'] == 'BLA'][
        'Spines percentage'].values
ca3_points = spines_percentages_dataframe[
    spines_percentages_dataframe['Input type'] == 'CA3'][
        'Spines percentage'].values

# Iterate over points and draw lines connecting them
for bla, ca3 in zip(bla_points, ca3_points):
    spines_percentages.ax.plot(
        [0.3, 1.3], [bla, ca3], color='gray', linestyle='--', linewidth=1)

# spines_percentages.save_figure(
#     saving_path / 'spines_percentages.png',
#     dpi=600)


# %% n event-based percentage of spines

BLA_percentages = {"Input type": "BLA"}
CA3_percentages = {"Input type": "CA3"}

for n_events in np.arange(0, 6):

    BLA_n_event_percentage = np.empty(len(segmenter_list))
    CA3_n_event_percentage = np.empty(len(segmenter_list))

    for n, segmenter in enumerate(segmenter_list):

        spines_with_n_BLA_event = sum(
            [1 if sum(flags) == n_events else 0
             for flags in segmenter.calcium_events_BLA])
        spines_with_n_CA3_event = sum(
            [1 if sum(flags) == n_events else 0
             for flags in segmenter.calcium_events_CA3])

        all_spines = len(segmenter.batch_spines_data)

        BLA_n_event_percentage[n] = spines_with_n_BLA_event / all_spines * 100
        CA3_n_event_percentage[n] = spines_with_n_CA3_event / all_spines * 100

    BLA_percentages[
        f"Spines percentage with {n_events} event(s)"] = (
            BLA_n_event_percentage)
    CA3_percentages[
        f"Spines percentage with {n_events} event(s)"] = (
            CA3_n_event_percentage)

n_event_percentage_dataframe = pd.DataFrame(
    [BLA_percentages, CA3_percentages])

event_colors = [
    ['#45818e', '#c06b58'],
    ['#46A4B9', '#B95B46'],
    ['#275c68', '#944838'],
    ['#1d454e', '#6f362a'],
    ['#132e34', '#4a241c'],
    ['#09171a', '#25120e']
]

fig = plt.figure()
gs = GridSpec(2, 2,
              width_ratios=[1, 0.5],
              height_ratios=[1, 1],
              hspace=0.05,
              wspace=0.3)

# Create the main plot in the larger upper part
ax_main = fig.add_subplot(gs[:, 0])
ax_main.grid(True, axis='y', linestyle='--', linewidth=1)
ax_main.set_axisbelow(True)
bar_width = 0.4

input_types = n_event_percentage_dataframe["Input type"]

bottom_BLA = np.zeros(len(input_types))
bottom_CA3 = np.zeros(len(input_types))

for i in range(6):
    # Get the current percentages for BLA and CA3
    event_column = f"Spines percentage with {i} event(s)"
    BLA_percentage = n_event_percentage_dataframe[event_column].iloc[0]
    CA3_percentage = n_event_percentage_dataframe[event_column].iloc[1]

    ax_main.bar(
        "BLA",
        BLA_percentage.mean(),
        bar_width,
        bottom=bottom_BLA,
        color=[event_colors[i][0]],
        edgecolor='black',
        clip_on=False)
    ax_main.bar(
        "CA3",
        CA3_percentage.mean(),
        bar_width,
        bottom=bottom_CA3,
        color=event_colors[i][1],
        edgecolor='black',
        clip_on=False)

    bottom_BLA += BLA_percentage.mean()
    bottom_CA3 += CA3_percentage.mean()


ax_main.spines[['top', 'right', 'bottom']].set_visible(False)
ax_main.spines[['left']].set_position(('outward', 20))
ax_main.set_ylim(0, 100)
ax_main.set_ylabel("Spine % with n events", fontsize=fontsize)
ax_main.tick_params(axis='both', which='major', labelsize=fontsize)

ax_zoom = fig.add_subplot(gs[0, 1])
ax_zoom.set_ylim(97, 100)  # Set the zoom-in range
ax_zoom.set_xlim(-0.5, 1.5)  # Focus on the two bars (BLA and CA3)

# Reset bottom for inset-like stacking
bottom_BLA_zoom = np.zeros(len(input_types))
bottom_CA3_zoom = np.zeros(len(input_types))

# Plot the inset-like zoomed in bars
for i in range(6):
    event_column = f"Spines percentage with {i} event(s)"
    BLA_percentage = n_event_percentage_dataframe[event_column].iloc[0]
    CA3_percentage = n_event_percentage_dataframe[event_column].iloc[1]

    # BLA bar in zoomed-in subplot
    ax_zoom.bar(
        "BLA",
        BLA_percentage.mean(),
        bar_width,
        bottom=bottom_BLA_zoom,
        color=event_colors[i][0],
        edgecolor='black',)

    # CA3 bar in zoomed-in subplot
    ax_zoom.bar(
        "CA3",
        CA3_percentage.mean(),
        bar_width,
        bottom=bottom_CA3_zoom,
        color=event_colors[i][1],
        edgecolor='black',)

    bottom_BLA_zoom += BLA_percentage.mean()
    bottom_CA3_zoom += CA3_percentage.mean()

# Customize the zoomed-in subplot
ax_zoom.spines[['top', 'right', 'bottom']].set_visible(False)
ax_zoom.grid(True, axis='y', linestyle='--', linewidth=0.5)
ax_zoom.set_xticklabels([])
ax_zoom.tick_params(axis='both', which='major', labelsize=fontsize)
ax_zoom.set_yticks(np.arange(97, 101, 1))

plt.subplots_adjust(
    left=0.2, right=0.95, top=0.90, bottom=0.1, wspace=0.8, hspace=0.05)


# %% Plot all morphology

fig, axes = plt.subplots(1, len(morphology_list))

for n, morph in enumerate(morphology_list):
    morph.plot(morph.neuron, ax=axes[n])

    axes[n].set_aspect('equal')
    for spine in ['top', 'bottom', 'left', 'right']:
        axes[n].spines[spine].set_visible(False)

    axes[n].set_ylabel('', fontsize=fontsize)
    axes[n].set_xlabel('', fontsize=fontsize)
    axes[n].tick_params(left=False, bottom=False, right=False, top=False)
    axes[n].set_xticklabels([])
    axes[n].set_yticklabels([])

    plt.tight_layout()

scalebar_length = 50
scalebar_position = (120, -100)
midpoint = scalebar_position[0] + (scalebar_length / 2)
axes[0].plot([scalebar_position[0], scalebar_position[0] + scalebar_length],
             [scalebar_position[1], scalebar_position[1]],
             color='black', lw=2, clip_on=False)
axes[0].text(midpoint, scalebar_position[1] - 70,
             f'{scalebar_length} μm',
             ha='center', va='bottom', fontsize=fontsize, clip_on=False)

# fig.savefig(
#     saving_path / 'all_morph.png',
#     dpi=600)

# %% Plot total dendritic length

dendritic_length = np.empty(len(segmenter_list))

for n, morph in enumerate(morphology_list):
    dendritic_length[n] = rp.analysis.stats.calculate_total_length(
        morph.neuron)

dendritic_length_dataframe = pd.DataFrame({
    "Dendritic Length": dendritic_length})

dendritic_length_plot = DataPlotter(
    figsize=single_group_figsize,
    font_size=fontsize,
    y_lims=(2000, 3500),
    y_label='Dendritic length (mm)',
    y_ticks=([2000, 2500, 3000, 3500]),
    y_pad=18,
    y_tick_labels=([2.0, 2.5, 3, 3.5]),
)

dendritic_length_plot.box_plot(
    dendritic_length_dataframe,
    x=None,
    y='Dendritic Length',
    box_width=single_box_width,
    boxprops=dict(facecolor='#808080'))

dendritic_length_plot.strip_plot(
    dendritic_length_dataframe,
    x=None,
    y='Dendritic Length',
    linewidth=1.5,
    edgecolor='#808080',
    fillcolor='none',
    jitter=0,
    strip_offset=single_group_strip_offset)

dendritic_length_plot.display_mean(
    dendritic_length_dataframe,
    x=None,
    y='Dendritic Length',
    offset=single_group_strip_offset,
    width=single_group_meanline_width,
    linewidth=3,
    color='black')

# dendritic_length_plot.save_figure(
#     saving_path / 'dendritic_length.png',
#     dpi=600)


# %% Overall spine number

spine_number = np.empty(len(segmenter_list))

for n, segmenter in enumerate(segmenter_list):
    spine_number[n] = len(segmenter.batch_spines_data)

spine_number_dataframe = pd.DataFrame({
    "Spine number": [int(n) for n in spine_number]})

spine_number_plot = DataPlotter(
    figsize=single_group_figsize,
    font_size=fontsize,
    y_lims=(500, 2000),
    y_label='Spine number',
    y_ticks=([500, 1000, 1500, 2000]),
    y_pad=18,
    y_tick_labels=([500, 1000, 1500, 2000]),
)

spine_number_plot.box_plot(
    spine_number_dataframe,
    x=None,
    y='Spine number',
    box_width=single_box_width,
    boxprops=dict(facecolor='#808080'))

spine_number_plot.strip_plot(
    spine_number_dataframe,
    x=None,
    y='Spine number',
    linewidth=1.5,
    edgecolor='#808080',
    fillcolor='none',
    jitter=0,
    strip_offset=single_group_strip_offset)

spine_number_plot.display_mean(
    spine_number_dataframe,
    x=None,
    y='Spine number',
    offset=single_group_strip_offset,
    width=single_group_meanline_width,
    linewidth=3,
    color='black')

# spine_number_plot.save_figure(
#     saving_path / 'overall_spine_number.png',
#     dpi=600)

# %% Ratio between overall spine number and dendritic length

fig, ax = plt.subplots(figsize=(4.2, 3))

ax.scatter(
    dendritic_length,
    spine_number,
    s=100,
    facecolors='none',
    edgecolors='#808080',
    lw=2)

slope, intercept = np.polyfit(dendritic_length, spine_number, 1)
line_x = np.array([dendritic_length.min(), dendritic_length.max()])
line_y = slope * line_x + intercept

ax.plot(
    line_x,
    line_y,
    color='black',
    ls='dotted',
    lw=1.5,)

ax.set_ylabel('Spine number', fontsize=fontsize)
ax.set_xlabel('Dendritic length (µm)', fontsize=fontsize)
ax.set_ylim(500, 2000)
ax.set_xlim(2000, 3500)

ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_position(('outward', 20))

plt.tight_layout()

# fig.savefig(
#     saving_path / 'spines_dendrites_ratio.png',
#     dpi=600)

# %% Mean and CI z score


def mean_and_ci(data, confidence=0.95):
    n = len(data)
    mean = np.mean(data, axis=0)
    sem = stats.sem(data, axis=0)
    ci = sem * stats.t.ppf((1 + confidence) / 2., n-1)
    return mean, ci


segmenter = segmenter_list[4]
ts = segmenter.batch_ts_BLA[4][0]

for input_type, is_event, color, input_str in zip(
        [segmenter.batch_z_scores_BLA,
         segmenter.batch_z_scores_CA3],
        [segmenter.calcium_events_BLA,
         segmenter.calcium_events_CA3],
        ['#46A4B9', '#B95B46'],
        ['_BLA.png', '_CA3.png']):

    event_zscores = np.array([
        input_type[i][j]
        for i in range(len(is_event))
        for j in range(len(is_event[i]))
        if is_event[i][j] == 1])

    eventless_zscores = np.array([
        input_type[i][j]
        for i in range(len(is_event))
        for j in range(len(is_event[i]))
        if is_event[i][j] == 0])

    for group, filename in zip([event_zscores, eventless_zscores],
                               ['zscore_with_ci_eventful',
                                'zscore_with_ci_eventless']):

        mean, ci = mean_and_ci(group)

        fig, ax = plt.subplots(figsize=(6, 3.5))
        ax.plot(
            ts,
            mean,
            color='black')

        ax.fill_between(
            ts,
            mean - ci,
            mean + ci,
            color=color,
            alpha=0.5)

        ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
        ax.set_ylim(-1, 2.)
        ax.set_xlim(0.5, 2.6)
        ax.set_xticks([])
        ax.set_yticks([])

        ax.plot([2., 2.5], [1., 1.], color='black', lw=1.5)
        ax.plot([2., 2.], [1., 1.5], color='black', lw=1.5)

        midpoint_h = (2. + 2.5) / 2
        ax.text(midpoint_h, 0.95,
                '500 ms', ha='center', va='top', fontsize=fontsize)

        midpoint_v = (1. + 1.5) / 2
        ax.text(2.05, midpoint_v,
                '0.5 z-score', ha='left', va='center', fontsize=fontsize)

        # filename_path = filename + input_str
        # fig.savefig(
        #     saving_path / filename_path,
        #     dpi=600)

# %% Z-score heatmap and trace

fontsize = 21

for input_type, is_event, color, input_str in zip(
        [segmenter.batch_z_scores_BLA, segmenter.batch_z_scores_CA3],
        [segmenter.calcium_events_BLA, segmenter.calcium_events_CA3],
        ['#46A4B9', '#B95B46'],
        ['BLA', 'CA3']):

    event_zscores = np.array([
        input_type[i][j]
        for i in range(len(is_event))
        for j in range(len(is_event[i]))
        if is_event[i][j] == 1])

    mean, ci = mean_and_ci(event_zscores)

    fig, axes = plt.subplots(2, 1, figsize=(5, 8), sharex=True,
                             gridspec_kw={'height_ratios': [1, 4]})

    axes[0].plot(
        ts,
        mean,
        color='black')

    axes[0].fill_between(
        ts,
        mean - ci,
        mean + ci,
        color=color,
        alpha=0.5)

    axes[0].spines[['top', 'bottom', 'right']].set_visible(False)
    axes[0].set_ylim(-1., 2.)
    axes[0].set_ylabel('Z-score', fontsize=fontsize)
    axes[0].tick_params(labelbottom=False, bottom=False)

    BLA = axes[1].imshow(
        event_zscores,
        aspect='auto',
        cmap='viridis',
        extent=[np.min(ts),
                np.max(ts),
                0,
                event_zscores.shape[0]],
        interpolation='none')

    for ax in axes:
        ax.tick_params(labelsize=fontsize)

    axes[1].set_ylabel('Sweeps', fontsize=fontsize)
    axes[1].set_xlabel('Time (s)', fontsize=fontsize)

    for ax in axes:
        ax.tick_params(labelsize=fontsize)

    plt.tight_layout()

    # fig.savefig(
    #     saving_path / f'zscore_heatmap_and_trace_{input_str}.png',
    #     dpi=600)

# %% Single branch with spines example

fontsize = 18

neurites = spynalyzer_list[2]
morphology = morphology_list[2]
sf = scanfields_list[2]
neurite = neurites[20]

fig, axes = plt.subplots(1, 3, figsize=(12, 5))

morphology.plot(morphology.neuron, ax=axes[0])
sf.plot(neurite.sf, ax=axes[0], edgecolor='red', linewidth=2)
axes[0].plot([80, 130], [-20, -20], color='black', lw=2)

neurite.plot(input_type='BLA', cmap=BLA_cmap, ax=axes[1], fontsize=fontsize)
neurite.plot(input_type='CA3', cmap=CA3_cmap, ax=axes[2], fontsize=fontsize)
for ax in axes:
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')
plt.tight_layout()

# fig.savefig(
#     saving_path / 'single_branch_with_spines.png',
#     dpi=600)

# %% Neurite-wise sequential interspine distance

input_types = {
    'BLA': {
        'color': '#46A4B9',
        'y_label': 'Sequential\nBLA-receiving spine \ndistance (µm)',
        'y_lims': (0, 80),
        'saving_name': "sequential_interspine_distance_BLA.png"
    },
    'CA3': {
        'color': '#B95B46',
        'y_label': 'Sequential\nCA3-receiving spine \ndistance (µm)',
        'y_lims': (0, 80),
        'saving_name': "sequential_interspine_distance_CA3.png"
    },
    'all': {
        'color': '#808080',
        'y_label': 'Sequential\nspine distance (µm)',
        'y_lims': (0, 40),
        'saving_name': "sequential_interspine_distance.png"
    }
}

all_distance_data = {}

for input_type, settings in input_types.items():
    all_neurons_data = []

    for neuron_number, neurites in enumerate(spynalyzer_list, start=1):
        all_neurites_data = []

        for n, neurite in enumerate(neurites):
            if neurite:
                distances = neurite.sequential_spine_distances(
                    input_type=input_type)

                interspine_distance_dataframe = pd.DataFrame({
                    "Neuron number": ([f"Neuron {neuron_number}"] *
                                      len(distances)),
                    "Neurite number": [f"Neurite {n}"] * len(distances),
                    "Distances": distances
                })

                all_neurites_data.append(interspine_distance_dataframe)

        neuron_distances_dataframe = pd.concat(
            all_neurites_data, ignore_index=True)
        all_neurons_data.append(neuron_distances_dataframe)

    distances_neuronwise_dataframe = pd.concat(
        all_neurons_data, ignore_index=True)

    averaged_distances = distances_neuronwise_dataframe.groupby(
        ["Neuron number", "Neurite number"])[
            "Distances"].mean().reset_index()

    averaged_distances.rename(
        columns={"Distances": "Average Distance"}, inplace=True)

    all_distance_data[input_type] = averaged_distances


for input_type, settings in input_types.items():

    averaged_distances = all_distance_data[input_type]

    sequential_interspine_distance_plot = DataPlotter(
        figsize=(5, 4),
        font_size=fontsize,
        y_lims=settings['y_lims'],
        x_lims=(0, 4),
        x_ticks=(np.arange(4)),
        x_tick_labels=([f"Neuron\n{i}" for i in np.arange(1, 5)]),
        y_label=settings['y_label'],
    )

    sequential_interspine_distance_plot.box_plot(
        averaged_distances,
        x="Neuron number",
        y="Average Distance",
        box_width=0.08,
        hue="Neuron number",
        color=settings['color']
    )

    sequential_interspine_distance_plot.strip_plot(
        averaged_distances,
        x="Neuron number",
        y="Average Distance",
        linewidth=1.5,
        jitter=0,
        edgecolor='#808080',
        fillcolor='none',
        strip_offset=0.3
    )

    sequential_interspine_distance_plot.display_mean(
        averaged_distances,
        x="Neuron number",
        y="Average Distance",
        offset=0.3,
        width=0.1,
        linewidth=3,
        color='black'
    )

    # sequential_interspine_distance_plot.save_figure(
    #     saving_path / settings['saving_name'],
    #     dpi=600)


# %% Input-wise spine 3d sholl analysis

for input_type, cmap_color, input_label in zip(
        ['BLA', 'CA3'],
        ['#46A4B9', '#B95B46'],
        ['_BLA', '_CA3']):

    fig, ax = plt.subplots()
    fig_sholl, ax_sholl = plt.subplots(figsize=(10, 4))

    all_apical = []
    all_basal = []

    for n, (morphology, segmenter) in enumerate(
            zip(morphology_list, segmenter_list)):

        apical_count, basal_count, radii = segmenter.sholl(
            morphology,
            50, 5,
            input_type,
            ax=ax,
            ax_sholl_curve=ax_sholl,
            countline_color='#808080',
            countline_alpha=0.9)

        morphology.plot(morphology.neuron, ax=ax)

        all_apical.append(apical_count)
        all_basal.append(basal_count)

    mean_apical, ci_apical = mean_and_ci(all_apical)
    mean_basal, ci_basal = mean_and_ci(all_basal)

    ax_sholl.plot(radii, mean_apical, color='black', lw=2)
    ax_sholl.plot(-radii, mean_basal, color='black', lw=2)

    ax_sholl.fill_between(
        radii, mean_apical - ci_apical, mean_apical + ci_apical,
        color=cmap_color, alpha=0.3, clip_on=False)

    ax_sholl.fill_between(
        -radii, mean_basal - ci_basal, mean_basal + ci_basal,
        color=cmap_color, alpha=0.3, clip_on=False)

    ax.set_aspect('equal')
    for spine in ['top', 'bottom', 'left', 'right']:
        ax.spines[spine].set_visible(False)

    for spine in ['top', 'right']:
        ax_sholl.spines[spine].set_visible(False)
    for spine in ['bottom', 'left']:
        ax_sholl.spines[spine].set_position(('outward', 20))

    ax.set_ylabel('', fontsize=fontsize)
    ax.set_xlabel('', fontsize=fontsize)
    ax.tick_params(left=False, bottom=False, right=False, top=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])

    ax_sholl.set_xlim(-250, 250)
    ax_sholl.set_xticks(np.linspace(-250, 250, 6))
    ax_sholl.set_xticklabels(np.linspace(-250, 250, 6, dtype=int),
                             fontsize=fontsize)

    ax_sholl.set_ylim(0, 120)
    ax_sholl.set_yticks(np.arange(0, 160, 40))
    ax_sholl.set_yticklabels(np.arange(0, 160, 40), fontsize=fontsize)
    ax_sholl.set_ylabel('Spine count', fontsize=fontsize)
    ax_sholl.set_xlabel('Radial distance from soma (µm)', fontsize=fontsize)

    plt.tight_layout()

    # fig_sholl.savefig(
    #     saving_path / f'sholl_curve{input_label}.png',
    #     dpi=600)

# %% Overall spines 3d sholl analysis

fig, ax = plt.subplots()
fig_sholl, ax_sholl = plt.subplots(figsize=(10, 4))

all_apical = []
all_basal = []

for n, (morphology, segmenter) in enumerate(
        zip(morphology_list, segmenter_list)):

    apical_count, basal_count, radii = segmenter.sholl(
        morphology,
        50, 5,
        ax=ax,
        ax_sholl_curve=ax_sholl,
        countline_color='#808080',
        countline_alpha=0.9)

    morphology.plot(morphology.neuron, ax=ax)

    all_apical.append(apical_count)
    all_basal.append(basal_count)

mean_apical, ci_apical = mean_and_ci(all_apical)
mean_basal, ci_basal = mean_and_ci(all_basal)

ax_sholl.plot(radii, mean_apical, color='black', lw=2)
ax_sholl.plot(-radii, mean_basal, color='black', lw=2)

ax_sholl.fill_between(
    radii, mean_apical - ci_apical, mean_apical + ci_apical,
    color='#B150BA', alpha=0.3, clip_on=False)

ax_sholl.fill_between(
    -radii, mean_basal - ci_basal, mean_basal + ci_basal,
    color='#B150BA', alpha=0.3, clip_on=False)

ax.set_aspect('equal')
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_visible(False)

for spine in ['top', 'right']:
    ax_sholl.spines[spine].set_visible(False)
for spine in ['bottom', 'left']:
    ax_sholl.spines[spine].set_position(('outward', 20))

ax.set_ylabel('', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(left=False, bottom=False, right=False, top=False)
ax.set_xticklabels([])
ax.set_yticklabels([])

ax_sholl.set_xlim(-250, 250)
ax_sholl.set_xticks(np.linspace(-250, 250, 6))
ax_sholl.set_xticklabels(np.linspace(-250, 250, 6, dtype=int),
                         fontsize=fontsize)

ax_sholl.set_ylim(0, 400)
ax_sholl.set_yticks(np.arange(0, 600, 200))
ax_sholl.set_yticklabels(np.arange(0, 600, 200), fontsize=fontsize)
ax_sholl.set_ylabel('Spine count', fontsize=fontsize)
ax_sholl.set_xlabel('Radial distance from soma (µm)', fontsize=fontsize)

plt.tight_layout()

# fig_sholl.savefig(
#     saving_path / 'sholl_curve_all_spines.png',
#     dpi=600)

# %% Sholl analyses of incrementally n_event spines

fontsize = 18

for n, (morphology, segmenter) in enumerate(
        zip(morphology_list, segmenter_list)):

    fig, axes = plt.subplots(2, 5, sharex=True, sharey=True, figsize=(15, 8))

    all_apical = []
    all_basal = []

    for row, region, cmap_color in zip(
            [0, 1], ['BLA', 'CA3'], ['#46A4B9', '#B95B46']):

        for n_event_threshold in range(5):

            try:

                apical_count, basal_count, radii = segmenter.sholl(
                    morphology,
                    50, 5,
                    region,
                    n_event_threshold=n_event_threshold,
                    ax_sholl_curve=axes[row][n_event_threshold],
                    countline_color='#808080',
                    countline_alpha=0.9)

                all_apical.append(apical_count)
                all_basal.append(basal_count)

                mean_apical, ci_apical = mean_and_ci(all_apical)
                mean_basal, ci_basal = mean_and_ci(all_basal)

                axes[row][n_event_threshold].plot(
                    radii, mean_apical, color='black', lw=2)
                axes[row][n_event_threshold].plot(
                    -radii, mean_basal, color='black', lw=2)

                axes[row][n_event_threshold].fill_between(
                    radii, mean_apical - ci_apical, mean_apical + ci_apical,
                    color=cmap_color, alpha=0.3, clip_on=False)

                axes[row][n_event_threshold].fill_between(
                    -radii, mean_basal - ci_basal, mean_basal + ci_basal,
                    color=cmap_color, alpha=0.3, clip_on=False)
            except IndexError:
                pass

    for ax_row in axes:
        for ax in ax_row:
            ax.set_aspect('equal')
            ax.spines[['top', 'right']].set_visible(False)
            ax.spines[['bottom', 'left']].set_position(('outward', 20))

            ax.set_ylabel('', fontsize=fontsize)
            ax.set_xlabel('', fontsize=fontsize)

            ax.set_xlim(-250, 250)
            ax.set_xticks(np.linspace(-250, 250, 6))
            ax.set_xticklabels(np.linspace(-250, 250, 6, dtype=int),
                               fontsize=fontsize)

            ax.set_ylim(0, 120)
            ax.set_yticks(np.arange(0, 160, 40))
            ax.set_yticklabels(np.arange(0, 160, 40), fontsize=fontsize)

    axes[0][0].set_ylabel('Spine count', fontsize=fontsize)
    axes[-1][-1].set_xlabel(
        'Radial distance from soma (µm)', fontsize=fontsize)

    plt.tight_layout()

# %% Sholl analysis drawing

morphology = morphology_list[2]
segmenter = segmenter_list[2]

fig, ax = plt.subplots(figsize=(10, 10))

segmenter.sholl(
    morphology,
    50, 5,
    # input_type='BLA',
    input_type='CA3',
    ax=ax,
    size=40,
    # cmap=BLA_cmap,
    cmap=CA3_cmap,
    fontsize=fontsize
    )

morphology.plot(morphology.neuron, ax=ax)

ax.set_aspect('equal')
for spine in ['top', 'bottom', 'left', 'right']:
    ax.spines[spine].set_visible(False)

ax.set_ylabel('', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(left=False, bottom=False, right=False, top=False)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()

# fig.savefig(
#     saving_path / 'sholl_drawing_CA3.png',
#     dpi=600)

# %% Branch degree color-coded morphology

fig = plt.figure()
gs = gridspec.GridSpec(2, len(morphology_list), height_ratios=[10, 0.2])
axes = [fig.add_subplot(gs[0, i]) for i in range(len(morphology_list))]

all_degrees = [node.branch_degree
               for morph in morphology_list
               for node in morph.neuron
               if node.branch_degree is not None]
vmin, vmax = min(all_degrees), max(all_degrees)


for i, morph in enumerate(morphology_list):

    cmap = plt.colormaps['viridis']

    degree_to_color = {degree: cmap((degree - vmin) / (vmax - vmin))
                       for degree in range(vmin, vmax + 1)}

    neurites = morph.neuron.neurite

    for neurite in neurites:

        branch_degree = neurite[0].branch_degree
        color = degree_to_color[branch_degree]
        morph.plot(neurite, ax=axes[i], color=color)

for ax in axes:
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_aspect('equal')


sm = plt.cm.ScalarMappable(
    cmap=cmap,
    norm=mpl.colors.Normalize(vmin=vmin, vmax=vmax))
sm.set_array([])

cbar_ax = fig.add_subplot(gs[1, :])
cbar = fig.colorbar(sm, cax=cbar_ax, orientation='horizontal', shrink=0.1)
cbar.set_label('Branch degree', fontsize=fontsize)
cbar.ax.set_xticks(np.arange(vmin, vmax + 1))
cbar.ax.tick_params(labelsize=fontsize)

plt.tight_layout()

# fig.savefig(
#     saving_path / 'branch_degree_pattern.png',
#     dpi=600)

# %% Spine number per branch degree

# segmenter = segmenter_list[2]
# morphology = morphology_list[2]

max_branch_degrees = max([node.branch_degree
                          for morphology in morphology_list
                          for node in morphology.neuron
                          if node.branch_degree is not None])

overall_spines_list = []
bla_spines_list = []
ca3_spines_list = []


for segmenter, morphology in zip(segmenter_list, morphology_list):

    spine_to_node_map_overall = {}
    spine_to_node_map_BLA = {}
    spine_to_node_map_CA3 = {}

    for spine, is_event_BLA, is_event_CA3 in zip(
            segmenter.batch_spines_data,
            segmenter.calcium_events_BLA,
            segmenter.calcium_events_CA3):

        spine_coords = [
            spine['centroid_fov'][0] *
            morphology.objective_resolution,
            spine['centroid_fov'][1] *
            morphology.objective_resolution,
            spine['roi_z']
        ]

        closest_node = None
        min_distance = float('inf')

        for node in morphology.neuron:
            node_coords = [node.x, node.y, node.z]
            distance = node_spine_distance(node_coords, spine_coords)

            if distance < min_distance:
                min_distance = distance
                closest_node_degree = node.branch_degree

        spine_to_node_map_overall[tuple(spine_coords)] = closest_node_degree
        if np.sum(is_event_BLA) > 0:
            spine_to_node_map_BLA[tuple(spine_coords)] = closest_node_degree
        if np.sum(is_event_CA3) > 0:
            spine_to_node_map_CA3[tuple(spine_coords)] = closest_node_degree

    unique_branch_degrees = np.unique(
        [node.branch_degree for node in morphology.neuron
         if node.branch_degree])

    branch_degree_wise_n_spines = np.zeros(
        len(unique_branch_degrees), dtype=int)
    branch_degree_wise_n_spines_BLA = np.zeros(
        len(unique_branch_degrees), dtype=int)
    branch_degree_wise_n_spines_CA3 = np.zeros(
        len(unique_branch_degrees), dtype=int)

    for spine_coords, branch_degree in spine_to_node_map_overall.items():
        branch_degree_wise_n_spines[branch_degree - 1] += 1

    for spine_coords, branch_degree in spine_to_node_map_BLA.items():
        branch_degree_wise_n_spines_BLA[branch_degree - 1] += 1

    for spine_coords, branch_degree in spine_to_node_map_CA3.items():
        branch_degree_wise_n_spines_CA3[branch_degree - 1] += 1

    spines_per_branch_df = pd.DataFrame({
        'Branch Degree': unique_branch_degrees,
        'Overall Spines': branch_degree_wise_n_spines,
        'BLA Spines': branch_degree_wise_n_spines_BLA,
        'CA3 Spines': branch_degree_wise_n_spines_CA3
    })

    total_overall = spines_per_branch_df['Overall Spines'].sum()
    total_bla = spines_per_branch_df['BLA Spines'].sum()
    total_ca3 = spines_per_branch_df['CA3 Spines'].sum()

    spines_per_branch_df_percentage = spines_per_branch_df.copy()

    spines_per_branch_df_percentage['Overall %'] = (
        spines_per_branch_df['Overall Spines'] / total_overall) * 100
    spines_per_branch_df_percentage['BLA %'] = (
        spines_per_branch_df['BLA Spines'] / total_bla) * 100
    spines_per_branch_df_percentage['CA3 %'] = (
        spines_per_branch_df['CA3 Spines'] / total_ca3) * 100

    overall_spines_list.append(
        np.pad(spines_per_branch_df_percentage['Overall %'].values,
               (0, max_branch_degrees -
                len(spines_per_branch_df_percentage['Overall %'])),
               'constant'))
    bla_spines_list.append(
        np.pad(spines_per_branch_df_percentage['BLA %'].values,
               (0, max_branch_degrees -
                len(spines_per_branch_df_percentage['BLA %'])),
               'constant'))
    ca3_spines_list.append(
        np.pad(spines_per_branch_df_percentage['CA3 %'].values,
               (0, max_branch_degrees -
                len(spines_per_branch_df_percentage['CA3 %'])),
               'constant'))

overall_spines_array = np.array(overall_spines_list)
bla_spines_array = np.array(bla_spines_list)
ca3_spines_array = np.array(ca3_spines_list)

columns = [f"Branch Degree {-~i}"
           for i in range(overall_spines_array.shape[1])]
overall_spines_df = pd.DataFrame(overall_spines_array, columns=columns)
bla_spines_df = pd.DataFrame(bla_spines_array, columns=columns)
ca3_spines_df = pd.DataFrame(ca3_spines_array, columns=columns)

overall_spines_df_melted = overall_spines_df.melt(
    var_name='Branch Degree', value_name='Spine Percentage').dropna()
bla_spines_df_melted = bla_spines_df.melt(
    var_name='Branch Degree', value_name='Spine Percentage').dropna()
ca3_spines_df_melted = ca3_spines_df.melt(
    var_name='Branch Degree', value_name='Spine Percentage').dropna()

# DataFrames, y_labels, colors, and filenames for each plot
spine_data = [
    (overall_spines_df_melted, overall_spines_df, 'Total\nspines/branch deg. %', '#B150BA', 'overall_spines_plot.png'),
    (bla_spines_df_melted, bla_spines_df, 'BLA\nspines/branch deg. %', '#46A4B9', 'bla_spines_plot.png'),
    (ca3_spines_df_melted, ca3_spines_df, 'CA3\nspines/branch deg. %', '#B95B46', 'ca3_spines_plot.png')
]

# Loop over each DataFrame and generate the plots
for data_melted, data_df, y_label, color, filename in spine_data:
    spine_per_branch_plot = DataPlotter(
        figsize=(4, 3),
        font_size=fontsize,
        y_lims=(0, 100),
        y_label=y_label,
        x_ticks=(np.arange(max_branch_degrees)),
        x_tick_labels=(np.arange(1, -~max_branch_degrees)),
        palette='viridis',
    )

    # Box plot
    spine_per_branch_plot.box_plot(
        data_melted,
        box_width=0.08,
        x='Branch Degree',
        y='Spine Percentage',
        color=color,
        hue='Branch Degree',
    )

    # Strip plot
    spine_per_branch_plot.strip_plot(
        data_melted,
        x='Branch Degree',
        y='Spine Percentage',
        linewidth=1.5,
        edgecolor='#808080',
        fillcolor='none',
        jitter=0,
        strip_offset=0.3
    )

    # Display the mean
    spine_per_branch_plot.display_mean(
        data_melted,
        x='Branch Degree',
        y='Spine Percentage',
        offset=0.3,
        width=0.1,
        linewidth=3,
        color='black'
    )

    # Plot neuron-wise lines
    for neuron_idx, neuron_data in data_df.iterrows():
        spine_per_branch_plot.ax.plot(
            np.array([0, 1, 2, 3]) + 0.3,
            neuron_data,
            linewidth=1,
            color='gray',
            alpha=0.7
        )

    spine_per_branch_plot.ax.text(x=1.5, y=-30, s='Branch degree',
                                  horizontalalignment='center')

    plt.tight_layout()

    spine_per_branch_plot.save_figure(saving_path / filename, dpi=600)


# %% I-F curve

fig, ax = plt.subplots(figsize=(4.5, 3))

for traces in traces_list:

    traces.plot_IF(ax=ax, line_marker=None, line_color='gray', line_alpha=0.3)

ax.plot(
    traces.I_steps,
    np.mean([traces.n_spikes for traces in traces_list], axis=0),
    color='black',
    lw=2)

ax.set_ylim(0, 20)
ax.set_yticks(np.linspace(0, 20, 5))

ax.set_xlim(0, 200)
ax.set_xticks(np.linspace(-100, 250, 8))

# %% Example recording with all synchronized signals

segmenter = segmenter_list[2]
traces = traces_list[2]
fontsize = 12
fig, axes = plt.subplots(4, 3, figsize=(20, 4),
                         width_ratios=[10, 1, 1],
                         height_ratios=[2, 1, 1, 1])

zoom_regions = [(1.25, 0.3), (6.30, 0.3)]

for row in range(4):
    axes[row][1].set_xlim(
        zoom_regions[0][0],
        zoom_regions[0][0] + zoom_regions[0][1])
    axes[row][2].set_xlim(
        zoom_regions[1][0],
        zoom_regions[1][0] + zoom_regions[1][1])
    axes[row][0].set_xlim(0, 10.)

    for col in [1, 2]:
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])

# Plot the recording
traces.data[0].setSweep(0, channel=0)

for column in range(3):
    axes[0][column].plot(
        traces.data[0].sweepX,
        traces.data[0].sweepY,
        color='black', lw=0.8)

    axes[0][column].set_yticks([])
    axes[0][column].set_xticks([])
    axes[0][column].spines[
        ['top', 'bottom', 'left', 'right']].set_visible(False)
    axes[0][column].spines[['left', 'bottom']].set_position(('outward', 20))

    axes[0][column].set_ylim(-500, 250)

# Add L-shaped scalebars
scalebar_x_length = [0.5, 0.1, 0.1]
scalebar_y_length = [200, 100, 100]
y_bar_text_dist = [0.1, 0.02, 0.02]
x_start = [9.5, 1.4, 6.5]
y_start = [-480, -480, -480]

for n, (xlen, ylen, ydist, xstart, ystart) in enumerate(zip(
        scalebar_x_length,
        scalebar_y_length,
        y_bar_text_dist,
        x_start,
        y_start)):

    axes[0][n].plot(
        [xstart, xstart + xlen], [ystart, ystart],
        color='black', clip_on=False)
    axes[0][n].plot(
        [xstart + xlen, xstart + xlen], [ystart, ystart + ylen],
        color='black', clip_on=False)
    axes[0][n].text(
        xstart + 0.5 * xlen,
        ystart - 200,
        f'{xlen} s', ha='center', fontsize=fontsize)
    axes[0][n].text(
        xstart + xlen + ydist,
        ystart + 0.5 * ylen,
        f'{ylen} pA', va='center', rotation=-90, fontsize=fontsize)

# Plot the LED D/A
traces.data[0].setSweep(0, channel=1)
led_y = [0. if i < 4.0 else 1. for i in traces.data[0].sweepY]

for column in range(3):
    axes[1][column].plot(traces.data[0].sweepX, led_y, color='#46A4B9')
    axes[1][column].set_ylim(0, 1)
    axes[1][column].set_yticks([0, 1])
    axes[1][column].set_xticks([])
    axes[1][column].set_yticklabels(['OFF', 'ON'])
    axes[1][column].tick_params(axis='y', labelsize=fontsize)
    axes[1][column].set_ylabel('LED', fontsize=fontsize, labelpad=10)
    axes[1][column].spines[['top', 'bottom', 'right']].set_visible(False)
    axes[1][column].spines[['left']].set_position(('outward', 20))


# Plot the electrode D/A
traces.data[0].setSweep(0, channel=2)
electrode_y = [0 if i < 4.0 else 1 for i in traces.data[0].sweepY]

for column in range(3):
    axes[2][column].plot(traces.data[0].sweepX, electrode_y, color='#B95B46')
    axes[2][column].set_ylim(0, 1)
    axes[2][column].set_yticks([0, 1])
    axes[2][column].set_xticks([])
    axes[2][column].set_yticklabels(['OFF', 'ON'])
    axes[2][column].tick_params(axis='y', labelsize=fontsize)
    axes[2][column].set_ylabel('Electr.', fontsize=fontsize, labelpad=10)
    axes[2][column].spines[['top', 'bottom', 'right']].set_visible(False)
    axes[2][column].spines[['left']].set_position(('outward', 20))

# Plot the 2p scanning windows
traces.data[0].setSweep(0, channel=4)

scan_trigger_indices = [i for i, element in enumerate(traces.data[0].sweepY)
                        if element > 4.0]

diffs = np.diff(scan_trigger_indices)

first_scan_start = scan_trigger_indices[0]
second_scan_start = scan_trigger_indices[np.where(diffs > 1)[0][0] + 1]

n_frames = 50
period = 0.0625
scan_duration_sec = n_frames * period
scan_duration_idx = int(scan_duration_sec * traces.data[0].sampleRate)

scan_y = np.zeros_like(traces.data[0].sweepX, dtype=int)
scan_y[first_scan_start:first_scan_start + scan_duration_idx] = 1
scan_y[second_scan_start:second_scan_start + scan_duration_idx] = 1

traces.data[0].setSweep(0, channel=3)
pmt_gate_indices = [i for i, element in enumerate(traces.data[0].sweepY)
                    if element > 4.0]
scan_y[pmt_gate_indices] = 0

for column in range(3):
    axes[3][column].fill_between(
        traces.data[0].sweepX,
        0, 1,
        where=scan_y == 1,
        color='green',
        alpha=0.3,
        edgecolor='none')
    axes[3][column].set_ylim(0, 1)
    axes[3][column].set_yticks([])
    axes[3][column].set_yticklabels([])
    axes[3][column].set_ylabel('2p scan', fontsize=fontsize, labelpad=40)

    axes[3][column].spines[
        ['top', 'bottom', 'right', 'left']].set_visible(False)
    axes[3][column].spines[['left']].set_position(('outward', 20))

axes[3][0].spines['bottom'].set_visible(True)
axes[3][0].set_xlim(0, 10.)
axes[3][0].tick_params(axis='x', labelsize=fontsize)
axes[3][0].set_xlabel('Time (s)', fontsize=fontsize)

for row in [1, 2, 3]:
    for col in [1, 2]:
        axes[row][col].spines[['left']].set_visible(False)
        axes[row][col].set_yticks([])
        axes[row][col].set_xticks([])
        axes[row][col].set_ylabel('')

plt.tight_layout()
plt.subplots_adjust(wspace=0.3, hspace=0.6)

# fig.savefig(
#     saving_path / 'recording_framework.png',
#     dpi=600)

# %% Traces of spines that are putatively activated by both CA3 and BLA

for segmenter in segmenter_list:

    BLA_spine_ind = [n for n, i in enumerate(segmenter.calcium_events_BLA)
                     if sum(i) > 0]

    CA3_spine_ind = [n for n, i in enumerate(segmenter.calcium_events_CA3)
                     if sum(i) > 0]

    common_indices = np.sort(
        list(set(BLA_spine_ind).intersection(CA3_spine_ind)))

    fig, axes = plt.subplots(1, 2, figsize=(7, 13), sharex=True, sharey=True)
    offset = 0
    increment = 2

    for i in common_indices:

        for sweep_n in range(len(segmenter.batch_ts_BLA[i])):

            axes[0].plot(
                segmenter.batch_ts_BLA[i][sweep_n],
                segmenter.batch_dFF_BLA[i][sweep_n] + offset,
                color='gray',
                alpha=0.3)

            axes[1].plot(
                segmenter.batch_ts_CA3[i][sweep_n],
                segmenter.batch_dFF_CA3[i][sweep_n] + offset,
                color='gray',
                alpha=0.3)

        axes[0].plot(
            segmenter.batch_ts_BLA[i][sweep_n],
            np.mean(segmenter.batch_dFF_BLA[i], axis=0) + offset,
            color='#46A4B9')

        axes[1].plot(
            segmenter.batch_ts_CA3[i][sweep_n],
            np.mean(segmenter.batch_dFF_CA3[i], axis=0) + offset,
            color='#B95B46')

        offset += increment

    for ax in axes:
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines[['bottom', 'left']].set_position(('outward', 20))

        ax.set_ylim(0, len(common_indices) * increment)
        ax.set_xlim(0, 4)
        y_ticks = np.linspace(
            0, (np.ceil(
                np.ceil(len(common_indices) * increment) / 10) * 10), 5)
        ax.set_yticks(y_ticks)
        ax.set_yticklabels((y_ticks / 2).astype(int))

        ax.axvline(1.0, color='red', alpha=0.3, lw=2.)

    axes[0].set_ylabel('Spines')
    axes[0].set_xlabel('Time (s)')
    plt.tight_layout()

# %%

segmenter = segmenter_list[2]

BLA_spine_ind = [n for n, i in enumerate(segmenter.calcium_events_BLA)
                 if sum(i) > 0]

CA3_spine_ind = [n for n, i in enumerate(segmenter.calcium_events_CA3)
                 if sum(i) > 0]

common_indices = np.sort(list(set(BLA_spine_ind).intersection(CA3_spine_ind)))

fig, ax = plt.subplots()

segmenter.plot_all_spines(ax=ax, spine_color='#bec0c2')

for i in common_indices:
    ax.scatter(
        segmenter.batch_spines_data[i]['centroid_fov'][0],
        segmenter.batch_spines_data[i]['centroid_fov'][1],
        color='red')

ax.set_aspect('equal')

ax.spines[['top', 'right', 'bottom', 'left']].set_visible(False)
ax.set_ylabel('', fontsize=fontsize)
ax.set_xlabel('', fontsize=fontsize)
ax.tick_params(left=False, bottom=False, right=False, top=False)
ax.set_xticklabels([])
ax.set_yticklabels([])

plt.tight_layout()

# %%

morphology = morphology_list[2]
stack = rp.Stack(neuronpaths_list[2])
sf = scanfields_list[2]


def normalize(image, min_value, max_value):
    min_value = max(min_value, np.min(image))
    max_value = min(max_value, np.max(image))
    norm_image = (image - min_value) / (max_value - min_value)

    return np.clip(norm_image, 0, 1)


fig, axes = plt.subplots(1, 4, figsize=(20, 5))

ax1 = fig.add_subplot(141, projection='3d')

vmin = -1000
vmax = 3000
z_layers = [8, 12, 17]

corners_deg = stack.corners_deg

x_min, y_min = corners_deg[0]
x_max, y_max = corners_deg[2]

for z_plane in z_layers:

    image = stack.image[z_plane]
    norm = Normalize(vmin=vmin, vmax=vmax)
    norm_image = normalize(image, vmin, vmax)
    framed_image = norm_image.copy()
    frame_width = 5
    framed_image[:frame_width, :] = vmax
    framed_image[-frame_width:, :] = vmax
    framed_image[:, :frame_width] = vmax
    framed_image[:, -frame_width:] = vmax

    x = np.linspace(x_min, x_max, image.shape[1])
    y = np.linspace(y_min, y_max, image.shape[0])
    Z = np.ones_like(image) * stack.zs[z_plane]
    X, Y = np.meshgrid(x, y)

    ax1.plot_surface(
        X, Y, Z,
        facecolors=plt.cm.binary_r(framed_image),
        rstride=1,
        cstride=1,
        shade=False,
        zorder=0)

    for rect in sf.neuComp[z_plane]:
        corners = np.array(
            [rect.bottom_left_deg,
             rect.bottom_right_deg,
             rect.top_right_deg,
             rect.top_left_deg])

        rect_corners_z = np.array(np.ones(4) * stack.zs[z_plane])
        ax1.add_collection3d(
            Poly3DCollection([
                list(zip(corners[:, 0], corners[:, 1], rect_corners_z))],
                             edgecolor='yellow', facecolor=None, zorder=2))

ax1.set_xticks([])
ax1.set_yticks([])
ax1.set_zticks([])

ax1.set_xticklabels([])
ax1.set_yticklabels([])
ax1.set_zticklabels([])

ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax1.set_zlabel('Z')

# Second, third, and fourth subplots: 2D plots
for n, z_layer in enumerate(z_layers):
    stack.plot(stack, z=z_layer, ax=axes[-~n], norm=(-1000, 3000))
    sf.plot(
        sf.neuComp[z_layer],
        ax=axes[-~n],
        scan_angle=True,
        edgecolor='yellow',
        linewidth=2)

# Titles for the plots
for ax in axes:
    ax.spines[['bottom', 'top', 'left', 'right']].set_visible(False)
    ax.tick_params(left=False, bottom=False, right=False, top=False)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.set_ylabel('', fontsize=fontsize)
    ax.set_xlabel('', fontsize=fontsize)
    ax.invert_yaxis()

plt.tight_layout(pad=2.0)

fig.savefig(
    saving_path / 'stack_and_rois.png',
    dpi=600)
