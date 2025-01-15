""" Created on Tue Oct 29 11:18:06 2024
    @author: dcupolillo """

import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from matplotlib_venn import venn2
import matplotlib.animation as animation
import random
from pathlib import Path
import seaborn as sns
import pandas as pd
import flammkuchen as fl
from dataplotter import DataPlotter
from spyne.core.utils.detect_event import (
    is_calcium_event_threshold_crossing, binarize_calcium_events_array)
from spyne.presentations.load_all_neurons import (
    prepare_spynalyzer_and_traces,
    prepare_classification)
from spyne.presentations.colormaps import BLA_cmap, CA3_cmap
from tqdm import tqdm
from sklearn.impute import SimpleImputer
import torch
import torch.nn as nn

# 1) visually check dually innervated spines masks and videos and traces
# 2) inspect unflagged traces for each method
#    compared to the mothod that gives the most


(
     spynalyzer_list,
     traces_list,
     morphology_list,
     scanfields_list,
     dataset_list,
     segmenter_list
) = prepare_spynalyzer_and_traces()

(
     paths,
     batch_zscores,
     batch_ts,
     zscore_data,
     event_probabilities,
     binary_event,
     dynamic_thresholds,
     difference_array
 ) = prepare_classification(dataset_list, segmenter_list)
\
saving_path = Path(
    r"Y:\dcupolillo\Lab meetings\Progress report\241204_PR15")
dataframe_path = Path(
    r"presentations\dataframe.h5")

if dataframe_path.is_file():
    dataframe = pd.DataFrame(fl.load(dataframe_path))

BLA_color = '#46A4B9'
CA3_color = '#B95B46'
gray_color = '#808080'
fontsize = 14
dpi = 600


# %% 01: Spine maps BLA/CA3
# Generates two side-by-side maps of segmented spines,
# overlaying "activated" BLA and CA3 spines for each neuron.

for n, (segmenter, morph) in enumerate(
        zip(segmenter_list, morphology_list)):

    fig, axes = plt.subplots(
        1, 2,
        figsize=(14, 12),
        sharex=True, sharey=True,
        layout="constrained")

    path = segmenter.dataset.folder
    title = f"{path.parts[-3]}_{path.parts[-2]}"
    fig.suptitle(title, fontsize=fontsize)

    for ax, region, cmap in zip(
            axes, ['BLA', 'CA3'], [BLA_cmap, CA3_cmap]):

        segmenter.plot_all_spines(
            spine_color='lightgray', ax=ax)

        segmenter.plot_events_spines(
            region, cmap=cmap, ax=ax,  fontsize=fontsize)

    for ax in axes:
        ax.set_aspect('equal')
        ax.spines[:].set_visible(False)

        ax.tick_params(
            left=False, bottom=False,
            right=False, top=False,
            labelleft=False, labelbottom=False)

    # fig.savefig(
    #     saving_path / f"{title}_spines_map_model_method.png",
    #     dpi=600)

# %% 02: Event probability distribution
# Plots the distribution of probability to contain an event
# and the 99th percentile cut-off
# for BLA/CA3 for all neurons

for n, segmenter in enumerate(segmenter_list):

    fig_hist, ax_hist = plt.subplots(
        figsize=(5, 4),
        layout="constrained")

    path = segmenter.dataset.folder
    title = f"{path.parts[-3]}_{path.parts[-2]}"
    fig_hist.suptitle(title, fontsize=fontsize)

    for region, color in zip(
            ["BLA", "CA3"], [BLA_color, CA3_color]):

        flat_probabilities = [
            sweep for spine in event_probabilities[n][region]
            for sweep in spine]

        sns.histplot(
            flat_probabilities,
            bins=20,
            color=color,
            kde=False,
            stat="probability",
            fill=False,
            element='step',
            ax=ax_hist,
        )

        ax_hist.spines[['top', 'right']].set_visible(False)
        ax_hist.spines[['left']].set_position(('outward', 20))

        ax_hist.set_xlabel("Calcium event probability", fontsize=fontsize)
        ax_hist.set_ylabel("Normalized frequency", fontsize=fontsize)
        ax_hist.set(
            xlabel="Calcium event probability",
            ylabel="Normalized frequency",
            xlim=(0.3, 0.7),
            ylim=(0, 0.2)
        )
        ax_hist.tick_params(axis='both', labelsize=fontsize)

        ax_hist.axvline(
            dynamic_thresholds[n][region], color=color, linestyle='dashed',
            label=(f"99th perc. {region} ="
                   f"{dynamic_thresholds[n][region]:.3f}")
        )

        legend = ax_hist.legend(
            loc='upper right',
            fontsize=fontsize,
        )

        # fig_hist.savefig(
        #     saving_path /
        # f"{title}_%Ca_event_probability_distrib.png", dpi=dpi)

# %% Detect event with threshold (2.5*SD)-crossing

for segmenter, dataset in zip(segmenter_list, dataset_list):

    save_dir = Path(dataset.folder.parent) / 'zscore_threshold_method_2.5'
    save_dir.mkdir(parents=True, exist_ok=True)

    if not ((save_dir / 'calcium_event_BLA.h5').exists() and
            (save_dir / 'calcium_event_CA3.h5').exists()):

        calcium_events_BLA = []
        calcium_events_CA3 = []

        for spine in tqdm(
                segmenter.batch_z_scores_BLA,
                desc="Analyzing BLA spines"):
            spine_flags_BLA = []
            for sweep in spine:
                spine_flags_BLA.append(
                    1 if is_calcium_event_threshold_crossing(
                        sweep,
                        threshold_factor=2.5) else 0)

            calcium_events_BLA.append(spine_flags_BLA)

        for spine in tqdm(
                segmenter.batch_z_scores_CA3,
                desc="Analyzing CA3 spines"):
            spine_flags_CA3 = []
            for sweep in spine:
                spine_flags_CA3.append(
                    1 if is_calcium_event_threshold_crossing(
                        sweep,
                        threshold_factor=2.5) else 0)

            calcium_events_CA3.append(spine_flags_CA3)

        fl.save(save_dir / 'calcium_event_BLA.h5', calcium_events_BLA)
        fl.save(save_dir / 'calcium_event_CA3.h5', calcium_events_CA3)

# %%

BLA_2_5_n = np.zeros((len(dataset_list)), dtype=int)
BLA_1_96_n = np.zeros((len(dataset_list)), dtype=int)
CA3_2_5_n = np.zeros((len(dataset_list)), dtype=int)
CA3_1_96_n = np.zeros((len(dataset_list)), dtype=int)

for n, (dataset, segmenter) in enumerate(zip(dataset_list, segmenter_list)):

    BLA_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_BLA.h5')

    BLA_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_BLA.h5')

    BLA_2_5_n[n] = sum(
        [1 if sum(spine) > 0 else 0
         for spine in BLA_2_5_crossing])
    BLA_1_96_n[n] = sum(
        [1 if sum(spine) > 0 else 0
         for spine in BLA_1_96_crossing])

    CA3_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_CA3.h5')

    CA3_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_CA3.h5')

    CA3_2_5_n[n] = sum(
        [1 if sum(spine) > 0 else 0
         for spine in CA3_2_5_crossing])
    CA3_1_96_n[n] = sum(
        [1 if sum(spine) > 0 else 0
         for spine in CA3_1_96_crossing])

threshold_crossing_df = pd.DataFrame(
    {'BLA_1.96*SD': BLA_1_96_n,
     'BLA_2.5*SD': BLA_2_5_n,
     'CA3_1.96*SD': CA3_1_96_n,
     'CA3_2.5*SD': CA3_2_5_n})

threshold_crossing_melted_df = threshold_crossing_df.melt(
    value_vars=['BLA_1.96*SD', 'BLA_2.5*SD', 'CA3_1.96*SD', 'CA3_2.5*SD'],
    var_name="Analysis",
    value_name="Spine detected")

threshold_crossing_plot = DataPlotter(
    figsize=(4, 3),
    font_size=fontsize,
    y_lims=(0, 400),
    y_label=r'N spines with Ca$^{+2}$ event',
    x_ticks=(np.arange(len(threshold_crossing_df.columns))),
    x_tick_labels=([
        'BLA\n1.96\nx S.D.',
        'BLA\n2.5\nx S.D.',
        'CA3\n1.96\nx S.D.',
        'CA3\n2.5\nx S.D.']))

threshold_crossing_plot.cloud_plot(
    threshold_crossing_melted_df,
    x="Analysis",
    y="Spine detected",
    hue="Analysis",
    cloud_offset=0.1,
    width=0.4,
    side='right',
    palette={
        "BLA_1.96*SD": BLA_color,
        "BLA_2.5*SD": BLA_color,
        "CA3_1.96*SD": CA3_color,
        "CA3_2.5*SD": CA3_color})

threshold_crossing_plot.strip_plot(
    threshold_crossing_melted_df,
    x="Analysis",
    y="Spine detected",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.0)

threshold_crossing_plot.display_mean(
    threshold_crossing_melted_df,
    x="Analysis",
    y="Spine detected",
    offset=0.0,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in threshold_crossing_df.iterrows():
    threshold_crossing_plot.ax.plot(
        [0, 1],
        [row["BLA_1.96*SD"], row["BLA_2.5*SD"]],
        color='lightgray',
        zorder=0
    )
    threshold_crossing_plot.ax.plot(
        [2, 3],
        [row["CA3_1.96*SD"], row["CA3_2.5*SD"]],
        color='lightgray',
        zorder=0
    )

threshold_crossing_plot.ax.grid(
    axis='y',
    color='lightgray',
    lw=1,
    ls='dashed',
    alpha=0.5,
    clip_on=False)

# threshold_crossing_plot.save_figure(
#     saving_path / 'threshold_crossing_1.96vs2.5_N.png',
#     dpi=dpi)

# %% Lost events when passing from 1.96 to 2.5 SD threshold

fig, axes = plt.subplots(
    3, 3,
    figsize=(6, 4.5),
    layout="constrained",
    sharex=True,
    sharey=True)

for n, ax in enumerate(axes.flat):
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 20))
    ax.set_ylim(-2, 4)
    ax.set_yticks([-2, 0, 4])
    ax.set_xlim(0, 4)
    ax.tick_params('both', labelsize=fontsize)

    if n >= len(segmenter_list):
        ax.axis('off')

fig.supylabel("Z-score", fontsize=fontsize)
axes[-1, 0].set_xlabel("Time (s)", fontsize=fontsize)

for n, (dataset, segmenter) in enumerate(zip(dataset_list, segmenter_list)):

    ax = axes.flat[n]

    path = segmenter.dataset.folder
    title = f"{path.parts[-3]}_{path.parts[-2]}"

    BLA_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_BLA.h5')

    BLA_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_BLA.h5')

    CA3_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_CA3.h5')

    CA3_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_CA3.h5')

    lost_events = []
    lost_events_ts = []
    sd = []

    for spine_n, spine_BLA in enumerate(BLA_1_96_crossing):

        if sum(spine_BLA) > 0 and sum(BLA_2_5_crossing[spine_n]) == 0:

            for sweep_n, (sweep, ts) in enumerate(zip(
                    segmenter.batch_z_scores_BLA[spine_n],
                    segmenter.batch_ts_BLA[spine_n])):

                if spine_BLA[sweep_n] == 1:
                    ax.plot(ts, sweep, color="lightgray",
                            clip_on=False)

                    lost_events.append(sweep)
                    lost_events_ts.append(ts)
                    sd.append(np.std(sweep))

    if lost_events:
        ax.plot(
            lost_events_ts[0],
            np.mean(lost_events, axis=0),
            color='green')
        ax.hlines(
            2.5 * np.mean(sd),
            xmin=lost_events_ts[0][16],
            xmax=lost_events_ts[0][21],
            ls="dashed",
            color='black',
            label="Mean\n2.5 x S.D.\nthresh.")
        ax.vlines(
            1.0,
            ymin=-3,
            ymax=-2,
            color='red',
            lw=2,
            clip_on=False)

# fig.savefig(
#     saving_path / "lost_events_from_196_to_25_threshold.png",
#     dpi=dpi)

# %%

BLA_2_5_perc = np.zeros((len(dataset_list)), dtype=int)
BLA_1_96_perc = np.zeros((len(dataset_list)), dtype=int)
CA3_2_5_perc = np.zeros((len(dataset_list)), dtype=int)
CA3_1_96_perc = np.zeros((len(dataset_list)), dtype=int)

for n, (dataset, segmenter) in enumerate(zip(dataset_list, segmenter_list)):

    BLA_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_BLA.h5')

    BLA_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_BLA.h5')

    BLA_2_5_perc[n] = (sum(
        [1 if sum(spine) > 0 else 0
         for spine in BLA_2_5_crossing]) /
        len(BLA_2_5_crossing)) * 100
    BLA_1_96_perc[n] = (sum(
        [1 if sum(spine) > 0 else 0
         for spine in BLA_1_96_crossing]) /
        len(BLA_1_96_crossing)) * 100

    CA3_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_CA3.h5')

    CA3_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_CA3.h5')

    CA3_2_5_perc[n] = (sum(
        [1 if sum(spine) > 0 else 0
         for spine in CA3_2_5_crossing]) /
        len(CA3_2_5_crossing)) * 100
    CA3_1_96_perc[n] = (sum(
        [1 if sum(spine) > 0 else 0
         for spine in CA3_1_96_crossing]) /
        len(CA3_1_96_crossing)) * 100

threshold_crossing_df = pd.DataFrame(
    {'BLA_1.96*SD': BLA_1_96_perc,
     'BLA_2.5*SD': BLA_2_5_perc,
     'CA3_1.96*SD': CA3_1_96_perc,
     'CA3_2.5*SD': CA3_2_5_perc})

threshold_crossing_melted_df = threshold_crossing_df.melt(
    value_vars=['BLA_1.96*SD', 'BLA_2.5*SD', 'CA3_1.96*SD', 'CA3_2.5*SD'],
    var_name="Analysis",
    value_name="Spine detected")

threshold_crossing_plot = DataPlotter(
    figsize=(4, 3),
    font_size=fontsize,
    y_lims=(0, 30),
    # y_ticks=([0, 25]),
    # y_tick_labels=([0, 25]),
    y_label=r'% spines with Ca$^{+2}$ event',
    x_ticks=(np.arange(len(threshold_crossing_df.columns))),
    x_tick_labels=([
        'BLA\n1.96\nx S.D.',
        'BLA\n2.5\nx S.D.',
        'CA3\n1.96\nx S.D.',
        'CA3\n2.5\nx S.D.']))

threshold_crossing_plot.cloud_plot(
    threshold_crossing_melted_df,
    x="Analysis",
    y="Spine detected",
    hue="Analysis",
    cloud_offset=0.1,
    width=0.4,
    side='right',
    palette={
        "BLA_1.96*SD": BLA_color,
        "BLA_2.5*SD": BLA_color,
        "CA3_1.96*SD": CA3_color,
        "CA3_2.5*SD": CA3_color})

threshold_crossing_plot.strip_plot(
    threshold_crossing_melted_df,
    x="Analysis",
    y="Spine detected",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.0)

threshold_crossing_plot.display_mean(
    threshold_crossing_melted_df,
    x="Analysis",
    y="Spine detected",
    offset=0.0,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in threshold_crossing_df.iterrows():
    threshold_crossing_plot.ax.plot(
        [0, 1],
        [row["BLA_1.96*SD"], row["BLA_2.5*SD"]],
        color='lightgray',
        zorder=0
    )
    threshold_crossing_plot.ax.plot(
        [2, 3],
        [row["CA3_1.96*SD"], row["CA3_2.5*SD"]],
        color='lightgray',
        zorder=0
    )

threshold_crossing_plot.ax.grid(
    axis='y',
    color='lightgray',
    lw=1.5,
    ls='dashed',
    alpha=0.5,
    clip_on=False)

# threshold_crossing_plot.save_figure(
#     saving_path / 'threshold_crossing_1.96vs2.5_percentage.png',
#     dpi=dpi)

# %% N of "dually innerv." spines 1.96 vs. 2.5 x S. D.

n_dual_2_5 = np.zeros((len(dataset_list)), dtype=int)
n_dual_1_96 = np.zeros((len(dataset_list)), dtype=int)

for n, (dataset, segmenter) in enumerate(zip(dataset_list, segmenter_list)):

    BLA_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_BLA.h5')

    BLA_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_BLA.h5')

    CA3_2_5_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method_2.5' /
        'calcium_event_CA3.h5')

    CA3_1_96_crossing = fl.load(
        Path(dataset.folder.parent) /
        'zscore_threshold_method' /
        'calcium_event_CA3.h5')

    n_dual_2_5[n] = sum(
        1 if (sum(spine_bla) > 0 and sum(spine_ca3) > 0) else 0
        for spine_bla, spine_ca3 in zip(BLA_2_5_crossing, CA3_2_5_crossing)
    )
    n_dual_1_96[n] = sum(
        1 if (sum(spine_bla) > 0 and sum(spine_ca3) > 0) else 0
        for spine_bla, spine_ca3 in zip(BLA_1_96_crossing, CA3_1_96_crossing)
    )

dual_spines_df = pd.DataFrame(
    {"n dual spines 1.96": n_dual_1_96,
     "n dual spines 2.5": n_dual_2_5})

dual_spines_melted_df = dual_spines_df.melt(
    value_vars=['n dual spines 1.96', 'n dual spines 2.5'],
    var_name="Analysis",
    value_name="N dual spines detected")

dual_spines_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label=r'N dual spines detected',
    x_ticks=([0, 1]),
    x_tick_labels=(['1.96\nx S.D.', '2.5\nx S.D.']),
)

dual_spines_plot.cloud_plot(
    dual_spines_melted_df,
    x="Analysis",
    y="N dual spines detected",
    cloud_offset=0.1,
    width=0.4,
    palette={
        "n dual spines 1.96": gray_color,
        "n dual spines 2.5": gray_color})

dual_spines_plot.strip_plot(
    dual_spines_melted_df,
    x="Analysis",
    y="N dual spines detected",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0)

dual_spines_plot.display_mean(
    dual_spines_melted_df,
    x="Analysis",
    y="N dual spines detected",
    offset=0,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in dual_spines_df.iterrows():
    dual_spines_plot.ax.plot(
        [0, 1],
        [row["n dual spines 1.96"], row["n dual spines 2.5"]],
        color='lightgray',
        zorder=0
    )

dual_spines_plot.ax.grid(
    axis='y',
    color='lightgray',
    lw=2,
    ls='dashed',
    alpha=0.5,
    clip_on=False)

dual_spines_plot.save_figure(
    saving_path / 'dual_spines_detected_1.96vs2.5_percentage.png',
    dpi=dpi)


# %% Example of a trace classified as event with 1.96 SD threshold

segmenter = segmenter_list[5]
dataset = dataset_list[5]

BLA_2_5_crossing = fl.load(
    Path(dataset.folder.parent) /
    'zscore_threshold_method_2.5' /
    'calcium_event_BLA.h5')

BLA_1_96_crossing = fl.load(
    Path(dataset.folder.parent) /
    'zscore_threshold_method' /
    'calcium_event_BLA.h5')

CA3_2_5_crossing = fl.load(
    Path(dataset.folder.parent) /
    'zscore_threshold_method_2.5' /
    'calcium_event_CA3.h5')

CA3_1_96_crossing = fl.load(
    Path(dataset.folder.parent) /
    'zscore_threshold_method' /
    'calcium_event_CA3.h5')

event_indices_1_96 = [
    n for n, spine in enumerate(BLA_1_96_crossing) if sum(spine) > 0]
event_indices_2_5 = [
    n for n, spine in enumerate(BLA_2_5_crossing) if sum(spine) > 0]

random.seed(42)
index = random.choice(event_indices_1_96)
z_scores = segmenter.batch_z_scores_BLA[index]
z_scores = np.stack(z_scores)
timestamps = segmenter.batch_ts_BLA[index]

sliced = z_scores[:, 16:21]
max_index = np.unravel_index(np.argmax(sliced), sliced.shape)
row_idx, col_idx = max_index

sd = np.std(z_scores[row_idx])
threshold = 1.96 * sd

fig, ax = plt.subplots(figsize=(5, 2), layout="constrained")
ax.plot(
        timestamps[row_idx],
        z_scores[row_idx],
        clip_on=False,
        color=BLA_color)
ax.spines[['top', 'right']].set_visible(False)
ax.spines['bottom'].set_position(("outward", 20))
ax.set_xlim(0, 4)
ax.set_ylim(-1, 4)
ax.set_yticks([-1, 4])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Z-score")
ax.hlines(
    threshold,
    xmin=timestamps[row_idx][16],
    xmax=timestamps[row_idx][21],
    color='green',
    ls='dashed')
ax.vlines(
    1.0,
    ymin=-2,
    ymax=-1,
    lw=2,
    color='red',
    clip_on=False,
    label="Stim.")
ax.axhline(0, color='lightgray', ls='dashed', zorder=0)
ax.text(
        x=timestamps[row_idx][21],
        y=2,
        s="1.96 x S.D.")
ax.legend(loc="best")

# fig.savefig(
#     saving_path / '1_96_event_example.png',
#     dpi=dpi)

# %% Example of a trace classified as event with 2.5 SD threshold

index = random.choice(event_indices_2_5)
z_scores = segmenter.batch_z_scores_BLA[index]
z_scores = np.stack(z_scores)
timestamps = segmenter.batch_ts_BLA[index]

sliced = z_scores[:, 16:21]
max_index = np.unravel_index(np.argmax(sliced), sliced.shape)
row_idx, col_idx = max_index

sd = np.std(z_scores[row_idx])
threshold = 2.5 * sd

fig, ax = plt.subplots(figsize=(5, 2), layout="constrained")
ax.plot(
        timestamps[row_idx],
        z_scores[row_idx],
        clip_on=False,
        color=BLA_color)
ax.spines[['top', 'right']].set_visible(False)
ax.spines['bottom'].set_position(("outward", 20))
ax.set_xlim(0, 4)
ax.set_ylim(-1, 4)
ax.set_yticks([-1, 4])
ax.set_xlabel("Time (s)")
ax.set_ylabel("Z-score")
ax.hlines(
    threshold,
    xmin=timestamps[row_idx][16],
    xmax=timestamps[row_idx][21],
    color='green',
    ls='dashed')
ax.vlines(
    1.0,
    ymin=-2,
    ymax=-1,
    lw=2,
    color='red',
    clip_on=False,
    label="Stim.")
ax.axhline(0, color='lightgray', ls='dashed', zorder=0)
ax.text(
        x=timestamps[row_idx][21],
        y=2,
        s="2.5 x S.D.")
ax.legend(loc="best")

# fig.savefig(
#     saving_path / '2_5_event_example.png',
#     dpi=dpi)

# %% "Dually innervated" spine z-score traces example

segmenter = segmenter_list[2]
dataset = dataset_list[2]

BLA_1_96_crossing = fl.load(
    Path(dataset.folder.parent) /
    'zscore_threshold_method' /
    'calcium_event_BLA.h5')

CA3_1_96_crossing = fl.load(
    Path(dataset.folder.parent) /
    'zscore_threshold_method' /
    'calcium_event_CA3.h5')

dual_indices = [
    n for n, (spine_bla, spine_ca3) in enumerate(zip(
        BLA_1_96_crossing, CA3_1_96_crossing))
    if (sum(spine_bla) > 0 and sum(spine_ca3) > 0)]

dual_spines = [
    (spine_bla, spine_ca3) for n, (spine_bla, spine_ca3) in enumerate(zip(
        BLA_1_96_crossing, CA3_1_96_crossing))
    if (sum(spine_bla) > 0 and sum(spine_ca3) > 0)]

examples_to_save = [
    27, 34]

for idx, dual_idx in enumerate(dual_indices):

    if not idx in examples_to_save:
        continue

    fig, axes = plt.subplots(
        5, 2,
        figsize=(5, 5),
        sharex=True,
        sharey=True,
        layout="constrained")

    for n, (sweep_BLA, sweep_CA3, ts) in enumerate(zip(
            segmenter.batch_z_scores_BLA[dual_idx],
            segmenter.batch_z_scores_CA3[dual_idx],
            segmenter.batch_ts_BLA[dual_idx])):

        sd_BLA = np.std(sweep_BLA)
        threshold_BLA = 1.96 * sd_BLA
        sd_CA3 = np.std(sweep_CA3)
        threshold_CA3 = 1.96 * sd_CA3

        axes[n][0].plot(ts, sweep_BLA, color=BLA_color)
        axes[n][1].plot(ts, sweep_CA3, color=CA3_color)
        axes[n][0].hlines(
            threshold_BLA,
            xmin=ts[16],
            xmax=ts[21],
            color='green',
            ls='dashed')
        axes[n][1].hlines(
            threshold_CA3,
            xmin=ts[16],
            xmax=ts[21],
            color='green',
            ls='dashed')

    for n, ax in enumerate(axes.flat):
        ax.set_ylim(-5, 5)
        ax.spines[["top", "right", "bottom"]].set_visible(False)
        ax.tick_params('x', length=0)
        ax.set_xticks([])
        ax.vlines(x=1., ymin=-4, ymax=-3., lw=3, color='red')

    for n_row, row in enumerate(axes):
        for n_col, col in enumerate(row):
            axes[n_row][n_col].text(x=0, y=4, s=str(
                dual_spines[idx][0 if n_col == 0 else 1][n_row]),
                color='red')

    axes[0][0].set_ylabel("Z-score")
    axes[-1][-1].hlines(-4, xmin=2.5, xmax=3., lw=2, color="black")
    axes[-1][-1].text(
        x=2.75, y=-4.5, s="500 ms",
        verticalalignment='top',
        horizontalalignment='center')

    # fig.savefig(
    #     saving_path / f"dual_spine_zscore_example_{idx}.png",
    #     dpi=dpi)

# %% Logistic regression prob. distributions

lr_model = fl.load(r"neuralnetwork\zscore_decoder\lr_model.h5")["lr_model"]

lr_probabilities_BLA = {}
lr_probabilities_CA3 = {}
imputer = SimpleImputer(strategy='mean')

fig_hist, ax_hist = plt.subplots(
    1, 2,
    figsize=(6, 3),
    sharex=True, sharey=True,
    layout="constrained")
for ax in ax_hist:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left']].set_position(('outward', 20))

    ax.set_xlabel("Calcium event probability", fontsize=fontsize)
    ax.set_ylabel("Normalized frequency", fontsize=fontsize)
    ax.set(
        xlabel="Calcium event prob.",
        ylabel="Normalized freq.",
        xlim=(0.0, 1.0),
        ylim=(0, 0.3)
    )
    ax.tick_params(axis='both', labelsize=fontsize)

fig_cdf, ax_cdf = plt.subplots(
    1, 2,
    figsize=(6, 3),
    sharex=True, sharey=True,
    layout="constrained")
for ax in ax_cdf:
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left']].set_position(('outward', 20))

    ax.set_xlabel("Calcium event probability", fontsize=fontsize)
    ax.set_ylabel("Cumulative frequency", fontsize=fontsize)
    ax.set(
        xlabel="Calcium event prob.",
        ylabel="Cumulative freq.",
        xlim=(0.0, 1.0),
        ylim=(0, 1.0)
    )
    ax.tick_params(axis='both', labelsize=fontsize)

for n, segmenter in enumerate(segmenter_list):

    num_spines = len(segmenter.batch_z_scores_BLA)
    num_sweeps = len(segmenter.batch_z_scores_BLA[0])

    event_probabilities_BLA = np.zeros((num_spines, num_sweeps))
    event_probabilities_CA3 = np.zeros((num_spines, num_sweeps))

    for n_spine, (spine_BLA, spine_CA3) in enumerate(zip(
            segmenter.batch_z_scores_BLA,
            segmenter.batch_z_scores_CA3)):

        for n_sweep, (sweep_BLA, sweep_CA3) in enumerate(zip(
                spine_BLA, spine_CA3)):

            if np.isnan(sweep_BLA).all():
                print(f"All NaN sweep at Spine {n_spine}, Sweep {n_sweep}."
                      "Skipping.")
                event_probabilities_BLA[n_spine, n_sweep] = 0.5
                continue
            if np.isnan(sweep_CA3).all():
                print(f"All NaN sweep at Spine {n_spine}, Sweep {n_sweep}."
                      "Skipping.")
                event_probabilities_CA3[n_spine, n_sweep] = 0.5
                continue

            if np.isnan(sweep_BLA).any():
                sweep_BLA = imputer.fit_transform(
                    sweep_BLA.reshape(1, -1)).flatten()
            if np.isnan(sweep_CA3).any():
                sweep_CA3 = imputer.fit_transform(
                    sweep_CA3.reshape(1, -1)).flatten()

            sweep_BLA = sweep_BLA.reshape(1, -1)
            event_probabilities_BLA[n_spine, n_sweep] = (
                lr_model.predict_proba(sweep_BLA)[0, 1])

            sweep_CA3 = sweep_CA3.reshape(1, -1)
            event_probabilities_CA3[n_spine, n_sweep] = (
                lr_model.predict_proba(sweep_CA3)[0, 1])

    lr_probabilities_BLA[f"Neuron_{n}"] = event_probabilities_BLA
    lr_probabilities_CA3[f"Neuron_{n}"] = event_probabilities_CA3

    flat_probabilities_BLA = [
        sweep for spine in event_probabilities_BLA
        for sweep in spine]
    flat_probabilities_CA3 = [
        sweep for spine in event_probabilities_CA3
        for sweep in spine]

    sns.histplot(
        flat_probabilities_BLA,
        bins=20,
        kde=False,
        color=BLA_color,
        stat="probability",
        fill=False,
        element='step',
        ax=ax_hist[0],
    )
    sns.histplot(
        flat_probabilities_CA3,
        bins=20,
        kde=False,
        color=CA3_color,
        stat="probability",
        fill=False,
        element='step',
        ax=ax_hist[1],
    )
    sns.ecdfplot(
        flat_probabilities_BLA,
        color=BLA_color,
        linewidth=2,
        label="BLA",
        ax=ax_cdf[0]
    )

    sns.ecdfplot(
        flat_probabilities_CA3,
        color=CA3_color,
        linewidth=2,
        label="CA3",
        ax=ax_cdf[1]
    )

# fig_hist.savefig(
#     saving_path / "lr_model_event_prob_distribution.png",
#     dpi=dpi)

# fig_cdf.savefig(
#     saving_path / "lr_model_event_prob_cdf.png",
#     dpi=dpi
# )

# %% LR prob. distributions - percentiles

desired_percentiles = [50, 70, 80, 90, 95, 99]

BLA_thresholds = np.zeros((len(segmenter_list), len(desired_percentiles)))
CA3_thresholds = np.zeros((len(segmenter_list), len(desired_percentiles)))

for n, (neuron, event_probability) in enumerate(
        lr_probabilities_BLA.items()):

    flat_probabilities_BLA = [
        sweep for spine in event_probability
        for sweep in spine
    ]
    for p, percentile in enumerate(desired_percentiles):
        threshold = np.percentile(flat_probabilities_BLA, percentile)

        BLA_thresholds[n, p] = threshold

for n, (neuron, event_probability) in enumerate(
        lr_probabilities_CA3.items()):

    flat_probabilities_CA3 = [
        sweep for spine in event_probability
        for sweep in spine
    ]
    for p, percentile in enumerate(desired_percentiles):
        threshold = np.percentile(flat_probabilities_CA3, percentile)

        CA3_thresholds[n, p] = threshold

BLA_thresholds_df = pd.DataFrame(
    data=BLA_thresholds,
    columns=desired_percentiles,
    index=[f"Neuron_{i}" for i in range(BLA_thresholds.shape[0])]
)
percentiles_BLA_df = BLA_thresholds_df.T

percentiles_BLA_melted = percentiles_BLA_df.reset_index().melt(
    id_vars="index",
    var_name="Neuron",
    value_name="Value")

percentiles_BLA_melted.rename(
    columns={"index": "Percentile"}, inplace=True)

lr_percentiles_plot = DataPlotter(
    figsize=(6, 3),
    font_size=fontsize,
    y_lims=(0, 1.),
    y_label='Probability',
    palette=[BLA_color, CA3_color],
    # x_ticks=([0, 1]),
    # x_tick_labels=(['BLA', 'CA3']),
)

# lr_percentiles_plot.box_plot(
#     percentiles_BLA_melted,
#     x="Percentile",
#     y="Value",
#     box_width=0.15,)

lr_percentiles_plot.cloud_plot(
    percentiles_BLA_melted,
    x="Percentile",
    y="Value",
    cloud_offset=0.1,
    width=0.4,
    side='right',
    )

lr_percentiles_plot.strip_plot(
    percentiles_BLA_melted,
    x="Percentile",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0)

lr_percentiles_plot.display_mean(
    percentiles_BLA_melted,
    x="Percentile",
    y="Value",
    offset=0,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in percentiles_BLA_df.T.iterrows():
    lr_percentiles_plot.ax.plot(
        [0, 1, 2, 3, 4, 5],
        row.values,
        color='lightgray',
        zorder=0
    )
lr_percentiles_plot.ax.text(
    2.5,
    -0.2,
    s="Percentiles",
    fontsize=fontsize,
    verticalalignment='center',
    horizontalalignment='center')

for line in np.arange(0, 1.2, 0.2):
    lr_percentiles_plot.ax.axhline(
        line, color='lightgray', ls='dashed', clip_on=False)
plt.tight_layout()

# lr_percentiles_plot.save_figure(
#     saving_path / "lr_percentiles.png", dpi=dpi)

# %% N "dual spines" for every percentile-cutoff

desired_percentiles = [50, 70, 80, 90, 95, 99]

results_BLA = []
results_CA3 = []
results_dual = []

for n, (neuron_BLA, probability_BLA) in enumerate(lr_probabilities_BLA.items()):
    neuron_CA3, probability_CA3 = list(lr_probabilities_CA3.items())[n]

    for p, (percentile_BLA, percentile_CA3) in enumerate(
        zip(BLA_thresholds[n], CA3_thresholds[n])
    ):

        binary_BLA = [[
                1 if sweep > percentile_BLA else 0
                for sweep in spine]
            for spine in probability_BLA]

        binary_CA3 = [[
            1 if sweep > percentile_CA3 else 0
            for sweep in spine
        ] for spine in probability_CA3]

        # dual_spines = sum(
        #     1 if any(bla and ca3 for bla, ca3 in zip(spine_BLA, spine_CA3))
        #     else 0
        #     for spine_BLA, spine_CA3 in zip(binary_BLA, binary_CA3)
        # )

        n_spines_BLA = sum(
            1 if sum(spine) > 0 else 0 for spine in binary_BLA)

        n_spines_CA3 = sum(
            1 if sum(spine) > 0 else 0 for spine in binary_CA3)

        dual_spines = sum(
            1 if sum(spine_BLA) > 0 and sum(spine_CA3) > 0 else 0 
            for spine_BLA, spine_CA3 in zip(binary_BLA, binary_CA3)
        )

        total_spines = len(segmenter_list[n].batch_spines_data)

        perc_spines_BLA = (n_spines_BLA / total_spines) * 100
        perc_spines_CA3 = (n_spines_CA3 / total_spines) * 100

        results_BLA.append({
            "Neuron": neuron_BLA,
            "Percentile": desired_percentiles[p],
            "Spines_with_Events": perc_spines_BLA
        })
        results_CA3.append({
            "Neuron": neuron_BLA,
            "Percentile": desired_percentiles[p],
            "Spines_with_Events": perc_spines_CA3
        })
        results_dual.append({
            "Neuron": neuron_BLA,
            "Percentile": desired_percentiles[p],
            "Dual_Spines": dual_spines
        })


results_BLA_df = pd.DataFrame(results_BLA)
results_CA3_df = pd.DataFrame(results_CA3)
dual_df = pd.DataFrame(results_dual)

lr_perc_spines_BLA_plot = DataPlotter(
    figsize=(5, 3),
    font_size=fontsize,
    y_lims=(0, 100),
    y_label=r'% BLA spines w/ Ca$^{+2}$ event',
)

lr_perc_spines_BLA_plot.cloud_plot(
    results_BLA_df,
    x="Percentile",
    y="Spines_with_Events",
    cloud_offset=0.1,
    width=0.4,
    side='right',
    color=BLA_color)

lr_perc_spines_BLA_plot.strip_plot(
    results_BLA_df,
    x="Percentile",
    y="Spines_with_Events",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0)

lr_perc_spines_BLA_plot.display_mean(
    results_BLA_df,
    x="Percentile",
    y="Spines_with_Events",
    offset=0,
    width=0.1,
    linewidth=3,
    color='black')

for neuron in results_BLA_df["Neuron"].unique():
    neuron_data = results_BLA_df[results_BLA_df["Neuron"] == neuron]

    neuron_data = neuron_data.sort_values(by="Percentile")

    lr_perc_spines_BLA_plot.ax.plot(
        np.arange(0, len(neuron_data['Percentile'])),
        neuron_data["Spines_with_Events"],
        color="lightgray",
        linewidth=1.5,
        zorder=0  # Send lines to the background
    )

lr_perc_spines_BLA_plot.ax.text(
    2.5,
    -20,
    s="Percentiles",
    fontsize=fontsize,
    verticalalignment='center',
    horizontalalignment='center')

for line in [25, 50, 75]:
    lr_perc_spines_BLA_plot.ax.axhline(
        line, color='lightgray', ls='dashed', clip_on=False)
plt.tight_layout()

# lr_perc_spines_BLA_plot.save_figure(
#     saving_path / 'lr_perc_BLA_spines_per_percentile.png',
#     dpi=dpi)

lr_dual_spines_plot = DataPlotter(
    figsize=(5, 3),
    font_size=fontsize,
    y_lims=(0, 2000),
    y_label=r'N dually activated spines',
)

lr_dual_spines_plot.box_plot(
    dual_df,
    x="Percentile",
    y="Dual_Spines",
    box_width=0.3,
    color=gray_color)

lr_dual_spines_plot.strip_plot(
    dual_df,
    x="Percentile",
    y="Dual_Spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

lr_dual_spines_plot.display_mean(
    dual_df,
    x="Percentile",
    y="Dual_Spines",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

for neuron in dual_df["Neuron"].unique():
    neuron_data = dual_df[dual_df["Neuron"] == neuron]

    neuron_data = neuron_data.sort_values(by="Percentile")

    lr_dual_spines_plot.ax.plot(
        np.arange(0, len(neuron_data['Percentile'])) + 0.3,
        neuron_data["Dual_Spines"],
        color="lightgray",
        linewidth=1.5,
        zorder=0  # Send lines to the background
    )

lr_dual_spines_plot.ax.text(
    2.5,
    -400,
    s="Percentiles",
    fontsize=fontsize,
    verticalalignment='center',
    horizontalalignment='center')

for line in np.linspace(0, 2000, 4):
    lr_dual_spines_plot.ax.axhline(
        line, color='lightgray', ls='dashed', clip_on=False, zorder=0)

plt.tight_layout()

# lr_dual_spines_plot.save_figure(
#     saving_path / 'lr_N_dual_spines_per_percentile.png',
#     dpi=dpi)

# %% Example of convolution

x = segmenter_list[-1].batch_ts_BLA[98][4]
y = segmenter_list[-1].batch_z_scores_BLA[98][4]

y_tensor = torch.tensor(y, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

# Define three different kernels using nn.Conv1d
kernels = {
    "Edge Detection":
        torch.tensor([[-1.0, 0.0, 1.0]], dtype=torch.float32).unsqueeze(0),
    "Smoothing":
        torch.tensor([[1.0, 1.0, 1.0]], dtype=torch.float32).unsqueeze(0) / 3,
    "Sharpening":
        torch.tensor([[-1.0, 2.0, -1.0]], dtype=torch.float32).unsqueeze(0),
}

feature_maps = {}
for name, kernel in kernels.items():
    conv = nn.Conv1d(
        in_channels=1,
        out_channels=1,
        kernel_size=3,
        padding=1,
        bias=False)
    conv.weight = nn.Parameter(kernel)
    feature_maps[name] = conv(y_tensor).squeeze().detach().numpy()

# Plot the results in a 2x3 grid
fig, ax = plt.subplots(
    2, 3,
    figsize=(8, 5),
    sharex=True,
    layout="constrained",
    gridspec_kw={"hspace": 0.2})

for i, (name, kernel) in enumerate(kernels.items()):
    ax[0, i].plot(
        x,
        y,
        color='blue')
    ax[0, i].set_ylabel('Z-Score', fontsize=fontsize)
    ax[0, i].set_ylim(-2, 6)

for i, ((name, feature_map), (_, kernel)) in enumerate(zip(
        feature_maps.items(), kernels.items())):
    ax[1, i].plot(
        x,
        feature_map,
        color='green',
        clip_on=False)
    kernel_str = ", ".join(f"{value:.1f}" for value in kernel.squeeze().tolist())
    ax[1, i].set_title(f"{name}\n[{kernel_str}]", fontsize=fontsize)
    ax[1, i].set_xlabel('Time (s)', fontsize=fontsize)
    ax[1, i].set_ylabel('Value', fontsize=fontsize)
    ax[1, i].set_ylim(-4, 4)

for i in range(3):
    ax[0, i].annotate(
        '',
        xy=(0.5, -.75),
        xycoords='axes fraction',
        xytext=(0.5, -0.5),
        textcoords='axes fraction',
        arrowprops=dict(facecolor='black', arrowstyle='-|>'),
        annotation_clip=False
    )

for ax in ax.flat:
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["bottom"].set_position(("outward", 20))
    ax.set_xlim(0, 4)
    ax.tick_params(labelsize=fontsize)

# fig.savefig(
#     saving_path / "1d_convolution_examples.png",
#     dpi=dpi)

# %% ReLU

x = np.linspace(-10, 10, 100)
y = np.maximum(0, x)

fig, ax = plt.subplots(figsize=(1.5, 1.2), layout="constrained")
ax.plot(x, y, color='red', lw=3)
ax.set_xlabel('Input')
ax.set_ylabel('Output')
ax.tick_params('y', length=0)
ax.set_xticks([0])
ax.set_yticks([])
ax.spines['left'].set_position('zero')
ax.spines['bottom'].set_position('zero')
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

fig.savefig(
    saving_path / "ReLU.png",
    dpi=dpi)

# %% Testing the new model (CNN)

fig_hist, ax_hist = plt.subplots(
    2, 4,
    figsize=(10, 4),
    sharex=True, sharey=True,
    layout="constrained")
ax_hist.flat[-1].set_axis_off()

for n, (segmenter, ax) in enumerate(zip(
        segmenter_list, ax_hist.flat[:-1])):

    BLA_prob = fl.load(
        Path(segmenter.dataset.folder.parent / 'new_calcium_event_BLA.h5'))
    CA3_prob = fl.load(
        Path(segmenter.dataset.folder.parent / 'new_calcium_event_CA3.h5'))

    path = segmenter.dataset.folder
    title = f"{path.parts[-3]}_{path.parts[-2]}"
    ax.set_title(title, fontsize=fontsize)

    for region, color in zip(
            [BLA_prob, CA3_prob], [BLA_color, CA3_color]):

        flat_probabilities = [
            sweep for spine in region
            for sweep in spine]

        sns.histplot(
            flat_probabilities,
            bins=20,
            color=color,
            kde=False,
            stat="probability",
            fill=False,
            element='step',
            ax=ax,
        )

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left']].set_position(('outward', 20))
    ax.set(
        xlim=(0., 1.),
        ylim=(0., 1.)
    )
    ax.tick_params(axis='both', labelsize=fontsize)
    ax.set_ylabel("")

    if n == 0:
        ax.set_xlabel("Calcium event prob.", fontsize=fontsize)
        ax.set_ylabel("Norm.\nfreq.", fontsize=fontsize)

# fig_hist.savefig(
#     saving_path / "new_model_probability_dist.png",
#     dpi=dpi)

# %%

BLA_spine_n_list = []
CA3_spine_n_list = []
BLA_threshold_list = []
CA3_threshold_list = []
dual_spines_list = []
spines_n = [
    len(segmenter.batch_spines_data) for segmenter in segmenter_list]

for n, segmenter in enumerate(segmenter_list):

    BLA_prob = fl.load(
        Path(segmenter.dataset.folder.parent / 'new_calcium_event_BLA.h5'))
    CA3_prob = fl.load(
        Path(segmenter.dataset.folder.parent / 'new_calcium_event_CA3.h5'))

    threshold_BLA, BLA_binary = binarize_calcium_events_array(BLA_prob, 99)
    threshold_CA3, CA3_binary = binarize_calcium_events_array(CA3_prob, 99)

    BLA_spine_n = sum([1 if sum(spine) > 0 else 0 for spine in BLA_binary])
    CA3_spine_n = sum([1 if sum(spine) > 0 else 0 for spine in CA3_binary])

    BLA_spine_n_list.append(BLA_spine_n)
    CA3_spine_n_list.append(CA3_spine_n)
    BLA_threshold_list.append(threshold_BLA)
    CA3_threshold_list.append(threshold_CA3)

    overlap_indices = []

    for idx, (bla_spine, ca3_spine) in enumerate(zip(
            BLA_binary, CA3_binary)):
        if sum(bla_spine) > 0 and sum(ca3_spine) > 0:
            overlap_indices.append(idx)
    dual_spines_list.append(len(overlap_indices))

n_spines_df = pd.DataFrame({
    "BLA": BLA_spine_n_list,
    "CA3": CA3_spine_n_list})
perc_spines_df = pd.DataFrame({
    "BLA %": [(bla / total) * 100
              for bla, total in zip(BLA_spine_n_list, spines_n)],
    "CA3 %": [(ca3 / total) * 100
              for ca3, total in zip(CA3_spine_n_list, spines_n)]})
dual_spines_df = pd.DataFrame({
    "dual_spines": dual_spines_list})

n_spines_melted = n_spines_df.melt(
    value_vars=['BLA', 'CA3'],
    var_name="Input",
    value_name="N spines")
perc_spines_melted = perc_spines_df.melt(
    value_vars=['BLA %', 'CA3 %'],
    var_name="Input",
    value_name="Percentage spines")
dual_spines_melted = dual_spines_df.melt(
    value_vars=['dual_spines'],
    var_name="Input",
    value_name="N spines")


n_spines_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 100),
    y_label=r'N spines with Ca$^{+2}$ event',
    x_ticks=([0, 1]),
    x_tick_labels=(['BLA', 'CA3']),
)

n_spines_plot.cloud_plot(
    n_spines_melted,
    x="Input",
    y="N spines",
    hue="Input",
    cloud_offset=0.2,
    width=0.4,
    palette={
        "BLA": BLA_color,
        "CA3": CA3_color})

n_spines_plot.strip_plot(
    n_spines_melted,
    x="Input",
    y="N spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0)

n_spines_plot.display_mean(
    n_spines_melted,
    x="Input",
    y="N spines",
    offset=0,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in n_spines_df.iterrows():
    n_spines_plot.ax.plot(
        [0, 1],
        [row["BLA"], row["CA3"]],
        color='lightgray',
        zorder=0
    )

# n_spines_plot.save_figure(
#     saving_path / 'n_spines_with_new_CNN.png',
#     dpi=dpi)

perc_spines_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 8),
    y_label=r'% spines with Ca$^{+2}$ event',
    x_ticks=([0, 1]),
    x_tick_labels=(['BLA', 'CA3']),
)

perc_spines_plot.cloud_plot(
    perc_spines_melted,
    x="Input",
    y="Percentage spines",
    hue="Input",
    cloud_offset=0.2,
    width=0.4,
    palette={
        "BLA %": BLA_color,
        "CA3 %": CA3_color})

perc_spines_plot.strip_plot(
    perc_spines_melted,
    x="Input",
    y="Percentage spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0)

perc_spines_plot.display_mean(
    perc_spines_melted,
    x="Input",
    y="Percentage spines",
    offset=0,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in perc_spines_df.iterrows():
    perc_spines_plot.ax.plot(
        [0, 1],
        [row["BLA %"], row["CA3 %"]],
        color='lightgray',
        zorder=0
    )

# perc_spines_plot.save_figure(
#     saving_path / 'perc_spines_with_new_CNN.png',
#     dpi=dpi)

dual_spines_plot = DataPlotter(
    figsize=(1.7, 3),
    font_size=fontsize,
    y_lims=(-5, 10),
    y_label=r"N 'dually activated' spines",
    x_ticks=([0, 1]),
    x_tick_labels=(['BLA', 'CA3']),
)

dual_spines_plot.cloud_plot(
    dual_spines_melted,
    x="Input",
    y="N spines",
    hue="Input",
    cloud_offset=0.2,
    width=0.4,
    color=gray_color,
    alpha=0.5)

dual_spines_plot.strip_plot(
    dual_spines_melted,
    x="Input",
    y="N spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0)

dual_spines_plot.display_mean(
    dual_spines_melted,
    x="Input",
    y="N spines",
    offset=0,
    width=0.1,
    linewidth=3,
    color='black')

dual_spines_plot.ax.axhline(
    0,
    color='lightgray',
    ls='dashed')

# dual_spines_plot.save_figure(
#     saving_path / 'dual_spines_with_new_CNN.png',
#     dpi=dpi)

# %%

segmenter = segmenter_list[6]
morphology = morphology_list[6]

BLA_prob = fl.load(
    Path(segmenter.dataset.folder.parent / 'new_calcium_event_BLA.h5'))
CA3_prob = fl.load(
    Path(segmenter.dataset.folder.parent / 'new_calcium_event_CA3.h5'))

threshold_BLA, BLA_binary = binarize_calcium_events_array(BLA_prob, 99)
threshold_CA3, CA3_binary = binarize_calcium_events_array(CA3_prob, 99)

fig, ax = plt.subplots(
    figsize=(10, 10),
    layout="constrained")

ax.spines[:].set_visible(False)
ax.tick_params('both', length=0)
ax.set(
       xticks=([]),
       yticks=([]),
       aspect='equal')

for n, spine in enumerate(segmenter.batch_spines_data):

    size = 30
    zorder = 1
    edgecolor = 'black'
    if sum(BLA_binary[n]) > 0 and sum(CA3_binary[n]) > 0:
        color = 'yellow'
    elif sum(BLA_binary[n]) > 0:
        color = BLA_color
    elif sum(CA3_binary[n]) > 0:
        color = CA3_color
    else:
        color = gray_color
        size = 10
        zorder = 0
        edgecolor = None

    ax.scatter(
        spine['centroid_fov'][0] * morphology.objective_resolution,
        spine['centroid_fov'][1] * morphology.objective_resolution,
        color=color,
        s=size,
        zorder=zorder,
        edgecolor=edgecolor)

fig.savefig(
    saving_path / "spine_map_with_new_CNN3.png",
    dpi=dpi)


# %% Christmas neuron

neuron_n = 5
segmenter = segmenter_list[neuron_n]
dataset = dataset_list[neuron_n]
analyzer = spynalyzer_list[neuron_n]
morph = morphology_list[neuron_n]
spines = [[spine['centroid_fov'][0] * morph.objective_resolution,
           spine['centroid_fov'][1] * morph.objective_resolution]
          for spine in segmenter.batch_spines_data]

# Constants
NUM_SPINES = 100  # Number of spines to pick in each iteration
TOTAL_FRAMES = 10  # Total number of frames in the animation
YELLOW_STAR_POSITION = (
    analyzer[6].neurite[-1].x,
    analyzer[6].neurite[-1].y)  # Position of the yellow star


random_subsamples = np.array(
    [random.sample(spines, NUM_SPINES) for _ in range(TOTAL_FRAMES)])
random_colors = np.array(
    [[random.choice(['#aa0000', '#f0db4d']) for _ in range(NUM_SPINES)]
     for _ in range(TOTAL_FRAMES)])


def init():
    ax.clear()
    ax.set_aspect('equal')
    ax.spines[['top', 'bottom', 'left', 'right']].set_visible(False)
    ax.tick_params('both', length=0)
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    morph.plot(morph.neuron, ax=ax, color='#075600', linewidth=5)
    ax.plot(
        YELLOW_STAR_POSITION[0],
        YELLOW_STAR_POSITION[1],
        marker='*', color='#bc943c', markersize=30)
    return ax,


def update(frame):
    ax.clear()
    init()
    selected_spines = random_subsamples[frame]
    selected_colors = random_colors[frame]
    for spine, color in zip(selected_spines, selected_colors):
        ax.plot(spine[0], spine[1], 'o', color=color, zorder=1, markersize=15)
    return ax,

fps=5

fig, ax = plt.subplots(figsize=(10, 10), layout="constrained")
fig.patch.set_facecolor('#282c34')
ax.set_facecolor('#282c34')

ani = animation.FuncAnimation(
    fig, update,
    frames=TOTAL_FRAMES,
    init_func=init,
    blit=False,
    repeat=True,
    interval=1000/fps)

ani.save(
    saving_path / 'christmas_animation.gif',
    writer='pillow',
    fps=fps)
