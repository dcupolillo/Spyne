""" Created on Wed Oct 30 13:40:05 2024
    @author: dcupolillo """

# %% Importing

import numpy as np
import flammkuchen as fl
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from scipy.stats import pearsonr, poisson, t, binom, binomtest, linregress
from dataplotter import DataPlotter
from spynalyzer.utils import node_spine_distance
import spynalyzer as spa
from spyne.core.utils.utils import mean_and_ci
from statistical_analysis.tests import (
    pairwise_comparison, group_comparison)
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd
from spyne.presentations.load_all_neurons import (
    prepare_spynalyzer_and_traces,
    prepare_classification)


saving_path = Path(r"Y:\SynEmo\Neurotalk_Dec24")
dataframe_path = Path(r"presentations\dataframe.h5")
BLA_color = '#46A4B9'
CA3_color = '#B95B46'
gray_color = '#808080'
fontsize = 14
dpi = 600

# %% List of plots generated in this file:

"""
01) N roi found
02) N of z planes
03) Total scannel area
04) Dendritic length (overall)
05) Dendritic length of apical vs basal dendrites

06) N of branches
07) N of branches per branching order
08) Neurite length per branch degree
09) Neurite length per branch degree in apical dendrites
10) Neurite length per branch degree in basal dendrites

11) N of spines total
12) N of spines vs. dendritic length (overall vs apical vs basal)
13) N of spines per compartment
14) N of spines per branch degree

15) % of "activated" spines BLA or CA3
16) % of "activated" spines BLA or CA3 in apical dendrites
17) % of "activated" spines BLA or CA3 in basal dendrites

18) % of "activated" spines BLA in apical vs basal dendrites
19) % of "activated" spines CA3 in apical vs basal dendrites

20) % of "activated" spines BLA per branch deg. apical vs basal dendrites
21) % of "activated" spines CA3 per branch deg. apical vs basal dendrites

22) % of "activated" spines BLA or CA3 per branch degree
23) % of "activated" spines BLA or CA3 per branch degree in apical dendrites
24) % of "activated" spines BLA or CA3 per branch degree in basal dendrites
25) % of "activated" spines BLA vs. CA3 per branch degree
26) % of "activated" spines branch-by-branch over total N of "activated" spines
27) % ratio of "activated" spines
    branch-by-branch over total "activated" spines

28) N of "activated" spines per compartment
29) N of "activated" spines per branch type

30) Density of spines per compartment
31) Density of spines per branch degree
32) Density of spines per branch degree in apical dendrites
33) Density of spines per branch degree in basal dendrites

34) Density of "activated" spines per compartment
35) Density of "activated" spines per branch degree
36) Density of "activated" spines per branch degree in apical dendrites
37) Density of "activated" spines per branch degree in basal dendrites

38) Z-scores traces of all neurons
39) Exploded view of all branches and spines

    % activated spines branch-by-branch - Andrea
    Have a look at masks
    N activation vs. spine size - Vince
    Probability map of spine activation - Mattia
    
"""


# %% Get data

if dataframe_path.is_file():
    dataframe = pd.DataFrame(fl.load(dataframe_path))

else:
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

    index_names = [None] * len(dataset_list)
    n_roi = [None] * len(segmenter_list)
    n_zplanes = [None] * len(segmenter_list)
    scanned_area = [None] * len(scanfields_list)
    dendritic_length = [None] * len(morphology_list)
    dendritic_length_apical = [None] * len(morphology_list)
    dendritic_length_basal = [None] * len(morphology_list)
    n_branches = [None] * len(morphology_list)
    n_branches_per_degree = [None] * len(morphology_list)
    branch_degrees_list = [None] * len(morphology_list)
    branch_degree_lengths = [None] * len(morphology_list)
    branch_degree_lenght_apical = [None] * len(morphology_list)
    branch_degree_lenght_basal = [None] * len(morphology_list)

    dyn_thresh_BLA = [None] * len(segmenter_list)
    dyn_thresh_CA3 = [None] * len(segmenter_list)
    n_spines = [None] * len(segmenter_list)
    n_spines_per_apical = [None] * len(segmenter_list)
    n_spines_per_basal = [None] * len(segmenter_list)
    n_spines_per_branch = [None] * len(segmenter_list)
    n_spines_per_branch_apical = [None] * len(segmenter_list)
    n_spines_per_branch_basal = [None] * len(segmenter_list)

    n_spines_BLA = [None] * len(segmenter_list)
    n_spines_CA3 = [None] * len(segmenter_list)
    n_spines_apical_BLA = [None] * len(segmenter_list)
    n_spines_apical_CA3 = [None] * len(segmenter_list)
    n_spines_basal_BLA = [None] * len(segmenter_list)
    n_spines_basal_CA3 = [None] * len(segmenter_list)

    n_spines_per_branch_BLA = [None] * len(segmenter_list)
    n_spines_per_branch_BLA_apical = [None] * len(segmenter_list)
    n_spines_per_branch_BLA_basal = [None] * len(segmenter_list)
    n_spines_per_branch_CA3 = [None] * len(segmenter_list)
    n_spines_per_branch_CA3_apical = [None] * len(segmenter_list)
    n_spines_per_branch_CA3_basal = [None] * len(segmenter_list)

    perc_spines_per_branch_BLA = [None] * len(segmenter_list)
    perc_spines_per_branch_CA3 = [None] * len(segmenter_list)
    perc_spines_per_branch_BLA_apical = [None] * len(segmenter_list)
    perc_spines_per_branch_CA3_apical = [None] * len(segmenter_list)
    perc_spines_per_branch_BLA_basal = [None] * len(segmenter_list)
    perc_spines_per_branch_CA3_basal = [None] * len(segmenter_list)

    for n, (segmenter, scanfields, morphology, dataset) in enumerate(zip(
                segmenter_list, scanfields_list,
                morphology_list, dataset_list)):

        path = dataset.folder
        index_names[n] = f"{path.parts[-3]}_{path.parts[-2]}"

        n_roi[n] = len(segmenter)
        n_zplanes[n] = len(scanfields.neuComp)
        scanned_area[n] = scanfields.neuComp.area / 1e3
        dendritic_length[n] = morphology.neuron.totlen() / 1e3
        dendritic_length_apical[n] = morphology.apical.totlen() / 1e3
        dendritic_length_basal[n] = morphology.basal.totlen() / 1e3

        n_branches[n] = len(morphology.neuron.neurite)
        branch_degree_lengths[n] = (
            morphology.neuron.neurite.cumulative_lengths)
        branch_degree_lenght_apical[n] = (
            morphology.apical.neurite.cumulative_lengths)
        branch_degree_lenght_basal[n] = (
            morphology.basal.neurite.cumulative_lengths)

        order_branches_list = []
        for i in np.arange(1, 5):
            order_branches = morphology.neuron.neurite.branches_degree.count(i)
            order_branches_list.append(order_branches)
        n_branches_per_degree[n] = order_branches_list
        n_spines[n] = len(segmenter.batch_spines_data)

        dynamic_threshold_BLA = np.percentile(
            np.array([sweep for spine in segmenter.calcium_events_BLA
                      for sweep in spine if not np.isnan(sweep)]), 99)
        dynamic_threshold_CA3 = np.percentile(
            np.array([sweep for spine in segmenter.calcium_events_CA3
                      for sweep in spine if not np.isnan(sweep)]), 99)

        dyn_thresh_BLA[n] = dynamic_threshold_BLA
        dyn_thresh_CA3[n] = dynamic_threshold_CA3

        event_binary_BLA = [
            [1 if sweep >= dynamic_threshold_BLA else 0 for sweep in spine]
            for spine in segmenter.calcium_events_BLA]
        n_spines_BLA[n] = sum(
            1 for spine in event_binary_BLA if sum(spine) > 0)
        event_binary_CA3 = [
            [1 if sweep >= dynamic_threshold_CA3 else 0 for sweep in spine]
            for spine in segmenter.calcium_events_CA3]
        n_spines_CA3[n] = sum(
            1 for spine in event_binary_CA3 if sum(spine) > 0)

        spine_to_node_map_overall = {}
        spine_to_node_map_BLA = {}
        spine_to_node_map_CA3 = {}

        apical_spines_count = 0
        basal_spines_count = 0
        apical_spines_count_BLA = 0
        basal_spines_count_BLA = 0
        apical_spines_count_CA3 = 0
        basal_spines_count_CA3 = 0

        for spine, is_event_BLA, is_event_CA3 in zip(
                segmenter.batch_spines_data,
                event_binary_BLA,
                event_binary_CA3):

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
                    closest_node_type = node.type

            spine_to_node_map_overall[
                tuple(spine_coords)] = (
                    closest_node_degree, closest_node_type)

            if np.sum(is_event_BLA) > 0:
                spine_to_node_map_BLA[tuple(spine_coords)] = (
                    closest_node_degree, closest_node_type)
            if np.sum(is_event_CA3) > 0:
                spine_to_node_map_CA3[tuple(spine_coords)] = (
                    closest_node_degree, closest_node_type)

            if closest_node_type == 'apical dendrite':
                apical_spines_count += 1
                if np.sum(is_event_BLA) > 0:
                    apical_spines_count_BLA += 1
                if np.sum(is_event_CA3) > 0:
                    apical_spines_count_CA3 += 1
            elif closest_node_type == 'basal dendrite':
                basal_spines_count += 1
                if np.sum(is_event_BLA) > 0:
                    basal_spines_count_BLA += 1
                if np.sum(is_event_CA3) > 0:
                    basal_spines_count_CA3 += 1

        unique_branch_degrees = np.unique(
            [node.branch_degree for node in morphology.neuron
             if node.branch_degree])

        branch_degrees_list[n] = unique_branch_degrees

        branch_degree_wise_n_spines = np.zeros(
            len(unique_branch_degrees), dtype=int)
        branch_degree_wise_n_spines_BLA = np.zeros(
            len(unique_branch_degrees), dtype=int)
        branch_degree_wise_n_spines_CA3 = np.zeros(
            len(unique_branch_degrees), dtype=int)

        branch_degree_wise_n_spines_apical = np.zeros(
            len(unique_branch_degrees), dtype=int)
        branch_degree_wise_n_spines_BLA_apical = np.zeros(
            len(unique_branch_degrees), dtype=int)
        branch_degree_wise_n_spines_CA3_apical = np.zeros(
            len(unique_branch_degrees), dtype=int)

        branch_degree_wise_n_spines_basal = np.zeros(
            len(unique_branch_degrees), dtype=int)
        branch_degree_wise_n_spines_BLA_basal = np.zeros(
            len(unique_branch_degrees), dtype=int)
        branch_degree_wise_n_spines_CA3_basal = np.zeros(
            len(unique_branch_degrees), dtype=int)

        for spine_coords, features in spine_to_node_map_overall.items():
            branch_degree, branch_type = features
            branch_degree_wise_n_spines[branch_degree - 1] += 1
            if branch_type == 'apical dendrite':
                branch_degree_wise_n_spines_apical[branch_degree - 1] += 1
            elif branch_type == 'basal dendrite':
                branch_degree_wise_n_spines_basal[branch_degree - 1] += 1

        for spine_coords, features in spine_to_node_map_BLA.items():
            branch_degree, branch_type = features
            branch_degree_wise_n_spines_BLA[branch_degree - 1] += 1
            if branch_type == 'apical dendrite':
                branch_degree_wise_n_spines_BLA_apical[branch_degree - 1] += 1
            elif branch_type == 'basal dendrite':
                branch_degree_wise_n_spines_BLA_basal[branch_degree - 1] += 1

        for spine_coords, features in spine_to_node_map_CA3.items():
            branch_degree, branch_type = features
            branch_degree_wise_n_spines_CA3[branch_degree - 1] += 1
            if branch_type == 'apical dendrite':
                branch_degree_wise_n_spines_CA3_apical[branch_degree - 1] += 1
            elif branch_type == 'basal dendrite':
                branch_degree_wise_n_spines_CA3_basal[branch_degree - 1] += 1

        n_spines_per_branch[n] = branch_degree_wise_n_spines
        n_spines_per_branch_BLA[n] = branch_degree_wise_n_spines_BLA
        n_spines_per_branch_CA3[n] = branch_degree_wise_n_spines_CA3
        perc_spines_per_branch_BLA[n] = list(
            (n_spine_BLA / n_spines) * 100
            if n_spine_BLA is not None and n_spines is not None
            and n_spines != 0 else 0
            for n_spine_BLA, n_spines in zip(
                branch_degree_wise_n_spines_BLA,
                branch_degree_wise_n_spines))
        perc_spines_per_branch_CA3[n] = list(
            (n_spine_CA3 / n_spines) * 100
            if n_spine_CA3 is not None and n_spines is not None
            and n_spines != 0 else 0
            for n_spine_CA3, n_spines in zip(
                branch_degree_wise_n_spines_CA3,
                branch_degree_wise_n_spines))

        n_spines_per_apical[n] = apical_spines_count
        n_spines_apical_BLA[n] = apical_spines_count_BLA
        n_spines_apical_CA3[n] = apical_spines_count_CA3

        n_spines_per_basal[n] = basal_spines_count
        n_spines_basal_BLA[n] = basal_spines_count_BLA
        n_spines_basal_CA3[n] = basal_spines_count_CA3

        n_spines_per_branch_apical[n] = branch_degree_wise_n_spines_apical
        n_spines_per_branch_BLA_apical[n] = (
            branch_degree_wise_n_spines_BLA_apical)
        n_spines_per_branch_CA3_apical[n] = (
            branch_degree_wise_n_spines_CA3_apical)
        perc_spines_per_branch_BLA_apical[n] = list(
            (n_spine_BLA / n_spines) * 100
            if n_spine_BLA is not None and n_spines is not None
            and n_spines != 0 else 0
            for n_spine_BLA, n_spines in zip(
                branch_degree_wise_n_spines_BLA_apical,
                branch_degree_wise_n_spines_apical))
        perc_spines_per_branch_CA3_apical[n] = list(
            (n_spine_CA3 / n_spines) * 100
            if n_spine_CA3 is not None and n_spines is not None
            and n_spines != 0 else 0
            for n_spine_CA3, n_spines in zip(
                branch_degree_wise_n_spines_CA3_apical,
                branch_degree_wise_n_spines_apical))

        n_spines_per_branch_basal[n] = branch_degree_wise_n_spines_basal
        n_spines_per_branch_BLA_basal[n] = (
            branch_degree_wise_n_spines_BLA_basal)
        n_spines_per_branch_CA3_basal[n] = (
            branch_degree_wise_n_spines_CA3_basal)
        perc_spines_per_branch_BLA_basal[n] = list(
            (n_spine_BLA / n_spines) * 100
            if n_spine_BLA is not None and n_spines is not None
            and n_spines != 0 else 0
            for n_spine_BLA, n_spines in zip(
                branch_degree_wise_n_spines_BLA_basal,
                branch_degree_wise_n_spines_basal))
        perc_spines_per_branch_CA3_basal[n] = list(
            (n_spine_CA3 / n_spines) * 100
            if n_spine_CA3 is not None and n_spines is not None
            and n_spines != 0 else 0
            for n_spine_CA3, n_spines in zip(
                branch_degree_wise_n_spines_CA3_basal,
                branch_degree_wise_n_spines_basal))

    dataframe = pd.DataFrame(
        {
            "N roi": n_roi,
            "N z-planes": n_zplanes,
            "Scanned area": scanned_area,
            "Dendritic length": dendritic_length,
            "Dendritic length apical": dendritic_length_apical,
            "Dendritic length basal": dendritic_length_basal,
            "N branches": n_branches,
            "N branch degree": n_branches_per_degree,
            "N spines": n_spines,
            "Dyn. thresh. BLA": dyn_thresh_BLA,
            "Dyn. thresh. CA3": dyn_thresh_CA3,
            "N spines BLA": n_spines_BLA,
            "N spines CA3": n_spines_CA3,
            "% spines BLA": [
                (n / total) * 100
                for n, total in zip(n_spines_BLA, n_spines)],
            "% spines apical BLA": [
                (n/total) * 100
                for n, total in zip(n_spines_apical_BLA, n_spines_per_apical)],
            "% spines basal BLA": [
                (n/total) * 100
                for n, total in zip(n_spines_basal_BLA, n_spines_per_basal)],
            "% spines CA3": [
                (n / total) * 100
                for n, total in zip(n_spines_CA3, n_spines)],
            "% spines apical CA3": [
                (n/total) * 100
                for n, total in zip(n_spines_apical_CA3, n_spines_per_apical)],
            "% spines basal CA3": [
                (n/total) * 100
                for n, total in zip(n_spines_basal_CA3, n_spines_per_basal)],
            "Branch degrees": branch_degrees_list,
            "Branch degree lengths": [
                [length / 1e3 for length in lengths]
                for lengths in branch_degree_lengths],
            "Branch degree lengths apical": [
                [length / 1e3 for length in lengths]
                for lengths in branch_degree_lenght_apical],
            "Branch degree lengths basal": [
                [length / 1e3 for length in lengths]
                for lengths in branch_degree_lenght_basal],
            "N spines per branch degree": n_spines_per_branch,
            "N spines per branch degree BLA": n_spines_per_branch_BLA,
            "N spines per branch degree CA3": n_spines_per_branch_CA3,
            "N spines per apical": n_spines_per_apical,
            "N spines per apical BLA": n_spines_apical_BLA,
            "N spines per apical CA3": n_spines_apical_CA3,
            "N spines per basal": n_spines_per_basal,
            "N spines per basal BLA": n_spines_basal_BLA,
            "N spines per basal CA3": n_spines_basal_CA3,
            "N spines per branch apical": n_spines_per_branch_apical,
            "N spines per branch apical BLA": n_spines_per_branch_BLA_apical,
            "N spines per branch apical CA3": n_spines_per_branch_CA3_apical,
            "N spines per branch basal": n_spines_per_branch_basal,
            "N spines per branch basal BLA": n_spines_per_branch_BLA_basal,
            "N spines per branch basal CA3": n_spines_per_branch_CA3_basal,
            "% spines per branch degree BLA": perc_spines_per_branch_BLA,
            "% spines per branch degree CA3": perc_spines_per_branch_CA3,
            "% spines per branch apical BLA": (
                perc_spines_per_branch_BLA_apical),
            "% spines per branch apical CA3": (
                perc_spines_per_branch_CA3_apical),
            "% spines per branch basal BLA": perc_spines_per_branch_BLA_basal,
            "% spines per branch basal CA3": perc_spines_per_branch_CA3_basal,
        },
        index=index_names
    )

    fl.save(dataframe_path, dataframe)

# %% 01) N roi found

n_roi_plot = DataPlotter(
    figsize=(1.7, 3),
    font_size=fontsize,
    y_lims=(100, 300),
    y_label='N of scanned ROIs',
)

n_roi_plot.box_plot(
    dataframe,
    x=None,
    y="N roi",
    box_width=0.3,
    color=gray_color,
    hue=None)

n_roi_plot.strip_plot(
    dataframe,
    x=None,
    y="N roi",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

n_roi_plot.display_mean(
    dataframe,
    x=None,
    y="N roi",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

plt.tight_layout()

# n_roi_plot.save_figure(
#     saving_path / 'n_rois.png',
#     dpi=dpi)


# %% 02) N of z planes

n_zplanes_plot = DataPlotter(
    figsize=(1.7, 3),
    font_size=fontsize,
    y_lims=(20, 60),
    y_label='N of scanned z-planes',
)

n_zplanes_plot.box_plot(
    dataframe,
    x=None,
    y="N z-planes",
    box_width=0.3,
    color=gray_color)

n_zplanes_plot.strip_plot(
    dataframe,
    x=None,
    y="N z-planes",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

n_zplanes_plot.display_mean(
    dataframe,
    x=None,
    y="N z-planes",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

plt.tight_layout()

# n_zplanes_plot.save_figure(
#     saving_path / 'n_zplanes.png',
#     dpi=dpi)

# %% 03) Total scannel area

scanned_area_plot = DataPlotter(
    figsize=(1.7, 3),
    font_size=fontsize,
    y_lims=(10, 30),
    y_label='Scanned area x 10³ (µm²)',
)

scanned_area_plot.box_plot(
    dataframe,
    x=None,
    y="Scanned area",
    box_width=0.3,
    color=gray_color)

scanned_area_plot.strip_plot(
    dataframe,
    x=None,
    y="Scanned area",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

scanned_area_plot.display_mean(
    dataframe,
    x=None,
    y="Scanned area",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

plt.tight_layout()

# scanned_area_plot.save_figure(
#     saving_path / 'scanned_area.png',
#     dpi=dpi)

# %% 04) Dendritic length (overall)

dendritic_length_plot = DataPlotter(
    figsize=(1.7, 3),
    font_size=fontsize,
    y_lims=(1.5, 3.5),
    y_label='Dendritic length (mm)',
)

dendritic_length_plot.box_plot(
    dataframe,
    x=None,
    y="Dendritic length",
    box_width=0.3,
    color=gray_color)

dendritic_length_plot.strip_plot(
    dataframe,
    x=None,
    y="Dendritic length",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

dendritic_length_plot.display_mean(
    dataframe,
    x=None,
    y="Dendritic length",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

plt.tight_layout()

# dendritic_length_plot.save_figure(
#     saving_path / 'dendritic_length.png',
#     dpi=dpi)

# %% 05) Dendritic length of apical vs. basal dendrites

melted_df = dataframe.melt(
    value_vars=["Dendritic length apical", "Dendritic length basal"],
    var_name="Compartment",
    value_name="Compartment-wise dendritic length")

dendritic_length_apical_vs_basal_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 3.),
    y_label='Dendritic length (mm)',
    x_ticks=([0, 1]),
    x_tick_labels=(["Apical", "Basal"])
)

dendritic_length_apical_vs_basal_plot.box_plot(
    melted_df,
    x="Compartment",
    y="Compartment-wise dendritic length",
    box_width=0.15,
    color=gray_color)

dendritic_length_apical_vs_basal_plot.strip_plot(
    melted_df,
    x="Compartment",
    y="Compartment-wise dendritic length",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

dendritic_length_apical_vs_basal_plot.display_mean(
    melted_df,
    x="Compartment",
    y="Compartment-wise dendritic length",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in dataframe.iterrows():
    dendritic_length_apical_vs_basal_plot.ax.plot(
        [0.3, 1.3],
        [row["Dendritic length apical"],
         row["Dendritic length basal"]],
        color='lightgray',
        zorder=0
    )

p_value = pairwise_comparison(
    melted_df[melted_df[
        'Compartment'] == 'Dendritic length apical'][
            'Compartment-wise dendritic length'],
    melted_df[melted_df[
        'Compartment'] == 'Dendritic length basal'][
            'Compartment-wise dendritic length'])

grouped_data = melted_df.groupby('Compartment').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Compartment', values='Compartment-wise dendritic length')

dendritic_length_apical_vs_basal_plot.annotate_stats(
    grouped_data,
    pairs=[("Dendritic length apical", "Dendritic length basal")],
    p_values=[0.00788],
    x_positions=[(0, 1)],
    offsets=[0],
    draw_feet=True,
    feet_length=0.05,
    height_offset=1,
    text_offset=2/100,)

plt.tight_layout()


# dendritic_length_apical_vs_basal_plot.save_figure(
#     saving_path / 'dendritic_length_apical_vs_basal.png',
#     dpi=dpi)

# %% 06) N of branches

n_branches_plot = DataPlotter(
    figsize=(1.7, 3),
    font_size=fontsize,
    y_lims=(15, 35),
    y_label='N of branches',
)

n_branches_plot.box_plot(
    dataframe,
    x=None,
    y="N branches",
    box_width=0.3,
    color=gray_color)

n_branches_plot.strip_plot(
    dataframe,
    x=None,
    y="N branches",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

n_branches_plot.display_mean(
    dataframe,
    x=None,
    y="N branches",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

plt.tight_layout()

# n_branches_plot.save_figure(
#     saving_path / 'n_branches.png',
#     dpi=dpi)

# %% 07) N of branches per branching order

max_degree = np.max(
    [len(entry) for entry in dataframe["N branch degree"].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

branch_degrees_df = pd.DataFrame(
    dataframe["N branch degree"].tolist(),
    columns=columns
)

melted_df = branch_degrees_df.melt(
    var_name="Branch degree",
    value_name="Value")

branches_order_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 30),
    y_label='N of branches',
)

branches_order_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=gray_color)

branches_order_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

branches_order_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

branches_order_plot.ax.text(
    y=-8, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

for n, row in branch_degrees_df.iterrows():
    branches_order_plot.ax.plot(
        np.arange(0.3, 4.3, 1),
        dataframe["N branch degree"].iloc[n],
        color='lightgray',
        zorder=0)

plt.tight_layout()

# branches_order_plot.save_figure(
#     saving_path / 'n_branches.png',
#     dpi=dpi)

# %% 08) Neurite length per branch degree

max_degree = np.max(
    [len(entry) for entry in dataframe["Branch degree lengths"].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

branch_degrees_length_df = pd.DataFrame(
    dataframe["Branch degree lengths"].tolist(),
    columns=columns
)

melted_df = branch_degrees_length_df.melt(
    var_name="Branch degree",
    value_name="Value")

branch_degrees_length_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 2.),
    y_label='Length (mm)',
)

branch_degrees_length_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=gray_color
)

branch_degrees_length_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

branch_degrees_length_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

branch_degrees_length_plot.ax.text(
    y=-0.5, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in branch_degrees_length_df.iterrows():

    y_values = [
        v for v in dataframe["Branch degree lengths"].iloc[n]
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    branch_degrees_length_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

plt.tight_layout()

# branch_degrees_length_plot.save_figure(
#     saving_path / 'branch_degrees_length.png',
#     dpi=dpi)

# %% 09) Neurite length per branch degree in apical dendrites

max_degree = np.max(
    [len(entry)
     for entry in dataframe["Branch degree lengths apical"].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

branch_degrees_length_df = pd.DataFrame(
    dataframe["Branch degree lengths apical"].tolist(),
    columns=columns
)

melted_df = branch_degrees_length_df.melt(
    var_name="Branch degree",
    value_name="Value")

branch_degrees_length_apical_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 2.),
    y_label='Length (mm)',
)

branch_degrees_length_apical_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=gray_color
)

branch_degrees_length_apical_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

branch_degrees_length_apical_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

branch_degrees_length_apical_plot.ax.text(
    y=-0.5, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row["Branch degree lengths apical"]
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    branch_degrees_length_apical_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

plt.tight_layout()

# branch_degrees_length_apical_plot.save_figure(
#     saving_path / 'branch_degrees_length_apical.png',
#     dpi=dpi)


# %% 10) Neurite length per branch degree in basal dendrites

max_degree = np.max(
    [len(entry)
     for entry in dataframe["Branch degree lengths basal"].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

branch_degrees_length_df = pd.DataFrame(
    dataframe["Branch degree lengths basal"].tolist(),
    columns=columns
)

melted_df = branch_degrees_length_df.melt(
    var_name="Branch degree",
    value_name="Value")

branch_degrees_length_basal_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 2.),
    y_label='Length (mm)',
)

branch_degrees_length_basal_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=gray_color
)

branch_degrees_length_basal_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

branch_degrees_length_basal_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

branch_degrees_length_basal_plot.ax.text(
    y=-0.5, x=1.,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row["Branch degree lengths basal"]
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    branch_degrees_length_basal_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

plt.tight_layout()

# branch_degrees_length_basal_plot.save_figure(
#     saving_path / 'branch_degrees_length_basal.png',
#     dpi=dpi)

# %% 11) N of spines total

n_spines_plot = DataPlotter(
    figsize=(1.7, 3),
    font_size=fontsize,
    y_lims=(500, 2000),
    y_label='N of spines',
)

n_spines_plot.box_plot(
    dataframe,
    x=None,
    y="N spines",
    box_width=0.3,
    color=gray_color)

n_spines_plot.strip_plot(
    dataframe,
    x=None,
    y="N spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

n_spines_plot.display_mean(
    dataframe,
    x=None,
    y="N spines",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

plt.tight_layout()

# n_spines_plot.save_figure(
#     saving_path / 'n_spines.png',
#     dpi=dpi)

# %% 12) N of spines vs. dendritic length

fig, axes = plt.subplots(1, 3, figsize=(8, 2), sharey=True, sharex=True)

spine_number = dataframe["N spines"]
dendritic_length = dataframe["Dendritic length"]

spine_number_apical = dataframe["N spines per apical"]
dendritic_length_apical = dataframe["Dendritic length apical"]

spine_number_basal = dataframe["N spines per basal"]
dendritic_length_basal = dataframe["Dendritic length basal"]

for n, (length, n_spines, color, ax) in enumerate(zip(
        [dendritic_length, dendritic_length_apical, dendritic_length_basal],
        [spine_number, spine_number_apical, spine_number_basal],
        [gray_color, 'red', 'blue'],
        axes)):

    ax.scatter(
        length,
        n_spines,
        s=60,
        facecolors='none',
        edgecolors=color,
        lw=2,
        clip_on=False)

    slope, intercept = np.polyfit(length, n_spines, 1)
    line_x = np.array([length.min(), length.max()])
    line_y = slope * line_x + intercept

    ax.plot(
        line_x,
        line_y,
        color='black',
        ls='dotted',
        lw=1.5,)

    r, p_value = pearsonr(length, n_spines)
    ax.text(
        0.05, 0.95, f"$r = {r:.2f}$",
        transform=ax.transAxes,
        ha="left", va="top",
        color=color,
        fontsize=fontsize
    )

    if n == 0:
        ax.set_ylabel('N spines', fontsize=fontsize)
        ax.set_xlabel('Dendritic length (mm)', fontsize=fontsize)
    else:
        ax.set_ylabel('', fontsize=fontsize)
        ax.set_xlabel('', fontsize=fontsize)
    ax.set_ylim(0, 2000)
    ax.set_xlim(0, 4)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_position(('outward', 20))

plt.tight_layout()

# fig.savefig(
#     saving_path / 'spines_dendrites_ratio.png',
#     dpi=dpi)

# %% 13) N of spines per compartment

spines_per_compartment_df = pd.DataFrame({
        'Apical': dataframe["N spines per apical"].tolist(),
        'Basal': dataframe["N spines per basal"].tolist()
    })

spines_per_compartment_melted_df = spines_per_compartment_df.melt(
    var_name="Compartment",
    value_name="N spines")


spines_per_compartment_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 1500),
    y_label='N of spines',
)

spines_per_compartment_plot.box_plot(
    spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    box_width=0.15,
    color=gray_color)

spines_per_compartment_plot.strip_plot(
    spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

spines_per_compartment_plot.display_mean(
    spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

x_values = [0.3, 1.3]

for n, row in spines_per_compartment_df.iterrows():

    y_values = [
        v for v in row
        if not pd.isna(v)]

    spines_per_compartment_plot.ax.plot(
        x_values,
        y_values,
        color='lightgray',
        zorder=0)

p_value = pairwise_comparison(
    dataframe["N spines per apical"].to_numpy(),
    dataframe["N spines per basal"].to_numpy())

grouped_data = spines_per_compartment_melted_df.groupby(
    'Compartment').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Compartment', values='N spines')

spines_per_compartment_plot.annotate_stats(
    grouped_data,
    pairs=[("Apical", "Basal")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[150],
    draw_feet=True,
    feet_length=30,
    height_offset=2,
    text_offset=2,)

plt.tight_layout()

# spines_per_compartment_plot.save_figure(
#     saving_path / 'spines_per_compartment.png',
#     dpi=dpi)

# %% 14) N of spines per branch degree

max_degree = np.max(
    [len(entry)
     for entry in dataframe['N spines per branch degree'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_degrees_df = pd.DataFrame(
    dataframe['N spines per branch degree'].tolist(),
    columns=columns
)

melted_df = spines_per_branch_degrees_df.melt(
    var_name="Branch degree",
    value_name="Value")

spines_per_branch_degree_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 1000),
    y_label='N of spines',
)

spines_per_branch_degree_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=gray_color)

spines_per_branch_degree_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

spines_per_branch_degree_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

spines_per_branch_degree_plot.ax.text(
    y=-200, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in spines_per_branch_degrees_df.iterrows():

    y_values = [
        v for v in dataframe["N spines per branch degree"].iloc[n]
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    spines_per_branch_degree_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

plt.tight_layout()

# spines_per_branch_degree_plot.save_figure(
#     saving_path / 'spine_number_per_branch_degree.png',
#     dpi=dpi)

# %% 15) % of "activated" spines BLA or CA3

melted_df = dataframe.melt(
    value_vars=["% spines BLA", "% spines CA3"],
    var_name="Input",
    value_name="Spine percentage")

perc_event_spines_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 8),
    y_label=r'% spines with Ca$^{+2}$ event',
    palette=[BLA_color, CA3_color],
    x_ticks=([0, 1]),
    x_tick_labels=(['BLA', 'CA3']),
)

perc_event_spines_plot.box_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    box_width=0.15,
    hue="Input")

perc_event_spines_plot.strip_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_event_spines_plot.display_mean(
    melted_df,
    x="Input",
    y="Spine percentage",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in dataframe.iterrows():
    perc_event_spines_plot.ax.plot(
        [0.3, 1.3],
        [row["% spines BLA"], row["% spines CA3"]],
        color='lightgray',
        zorder=0
    )

p_value = pairwise_comparison(
    melted_df[melted_df['Input'] == '% spines BLA']['Spine percentage'],
    melted_df[melted_df['Input'] == '% spines CA3']['Spine percentage'])

grouped_data = melted_df.groupby('Input').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Input', values='Spine percentage')

perc_event_spines_plot.annotate_stats(
    grouped_data,
    pairs=[("% spines BLA", "% spines CA3")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[0],
    draw_feet=True,
    feet_length=0.2,
    height_offset=2,
    text_offset=2/100,)

plt.tight_layout()

# perc_event_spines_plot.save_figure(
#     saving_path / 'perc_event_spineso.png',
#     dpi=dpi)

# %% 16) % of "activated" spines BLA or CA3 in apical dendrites

melted_df = dataframe.melt(
    value_vars=["% spines apical BLA", "% spines apical CA3"],
    var_name="Input",
    value_name="Spine percentage")

perc_event_spines_apical_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 8),
    y_label=r'% apical spines with Ca$^{+2}$ event',
    palette=[BLA_color, CA3_color],
    x_ticks=([0, 1]),
    x_tick_labels=(['BLA', 'CA3']),
)

perc_event_spines_apical_plot.box_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    box_width=0.15,
    hue="Input")

perc_event_spines_apical_plot.strip_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_event_spines_apical_plot.display_mean(
    melted_df,
    x="Input",
    y="Spine percentage",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in dataframe.iterrows():
    perc_event_spines_apical_plot.ax.plot(
        [0.3, 1.3],
        [row["% spines apical BLA"],
         row["% spines apical CA3"]],
        color='lightgray',
        zorder=0
    )

p_value = pairwise_comparison(
    melted_df[melted_df['Input'] == '% spines apical BLA']['Spine percentage'],
    melted_df[melted_df['Input'] == '% spines apical CA3']['Spine percentage'])

grouped_data = melted_df.groupby('Input').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Input', values='Spine percentage')

perc_event_spines_apical_plot.annotate_stats(
    grouped_data,
    pairs=[("% spines apical BLA", "% spines apical CA3")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[0],
    draw_feet=True,
    feet_length=0.2,
    height_offset=2,
    text_offset=2/100,)

plt.tight_layout()

# perc_event_spines_apical_plot.save_figure(
#     saving_path / 'perc_event_spines_apical.png',
#     dpi=dpi)

# %% 17) % of "activated" spines BLA or CA3 in basal dendrites

melted_df = dataframe.melt(
    value_vars=["% spines basal BLA", "% spines basal CA3"],
    var_name="Input",
    value_name="Spine percentage")

perc_event_spines_basal_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 8),
    y_label=r'% basal spines with Ca$^{+2}$ event',
    palette=[BLA_color, CA3_color],
    x_ticks=([0, 1]),
    x_tick_labels=(['BLA', 'CA3']),
)

perc_event_spines_basal_plot.box_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    box_width=0.15,
    hue="Input")

perc_event_spines_basal_plot.strip_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_event_spines_basal_plot.display_mean(
    melted_df,
    x="Input",
    y="Spine percentage",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in dataframe.iterrows():
    perc_event_spines_basal_plot.ax.plot(
        [0.3, 1.3],
        [row["% spines basal BLA"],
         row["% spines basal CA3"]],
        color='lightgray',
        zorder=0
    )

p_value = pairwise_comparison(
    melted_df[melted_df['Input'] == '% spines basal BLA']['Spine percentage'],
    melted_df[melted_df['Input'] == '% spines basal CA3']['Spine percentage'])

grouped_data = melted_df.groupby('Input').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Input', values='Spine percentage')

perc_event_spines_basal_plot.annotate_stats(
    grouped_data,
    pairs=[("% spines basal BLA", "% spines basal CA3")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[0],
    draw_feet=True,
    feet_length=0.2,
    height_offset=2,
    text_offset=2/100,)

plt.tight_layout()

# perc_event_spines_basal_plot.save_figure(
#     saving_path / 'perc_event_spines_basal.png',
#     dpi=dpi)

# %% 18) % of "activated" spines BLA in apical vs basal dendrites

melted_df = dataframe.melt(
    value_vars=["% spines apical BLA", "% spines basal BLA"],
    var_name="Input",
    value_name="Spine percentage")

perc_BLA_apical_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 8),
    y_label=r'% spines with Ca$^{+2}$ event',
    x_ticks=([0, 1]),
    x_tick_labels=(['Apical', 'Basal']),
)

perc_BLA_apical_plot.box_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    box_width=0.15,
    hue="Input",
    color=BLA_color)

perc_BLA_apical_plot.strip_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_BLA_apical_plot.display_mean(
    melted_df,
    x="Input",
    y="Spine percentage",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in dataframe.iterrows():
    perc_BLA_apical_plot.ax.plot(
        [0.3, 1.3],
        [row["% spines apical BLA"], row["% spines basal BLA"]],
        color='lightgray',
        zorder=0
    )

p_value = pairwise_comparison(
    melted_df[melted_df['Input'] == '% spines apical BLA']['Spine percentage'],
    melted_df[melted_df['Input'] == '% spines basal BLA']['Spine percentage'])

grouped_data = melted_df.groupby('Input').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Input', values='Spine percentage')

perc_BLA_apical_plot.annotate_stats(
    grouped_data,
    pairs=[("% spines apical BLA", "% spines basal BLA")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[0],
    draw_feet=True,
    feet_length=0.2,
    height_offset=2,
    text_offset=2/100,)

plt.tight_layout()

# perc_BLA_apical_plot.save_figure(
#     saving_path / 'perc_BLA_apical_plot.png',
#     dpi=dpi)

# %% 19) % of "activated" spines CA3 in apical vs basal dendrites

melted_df = dataframe.melt(
    value_vars=["% spines apical CA3", "% spines basal CA3"],
    var_name="Input",
    value_name="Spine percentage")

perc_CA3_apical_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 8),
    y_label=r'% spines with Ca$^{+2}$ event',
    x_ticks=([0, 1]),
    x_tick_labels=(['Apical', 'Basal']),
)

perc_CA3_apical_plot.box_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    box_width=0.15,
    hue="Input",
    color=CA3_color)

perc_CA3_apical_plot.strip_plot(
    melted_df,
    x="Input",
    y="Spine percentage",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_CA3_apical_plot.display_mean(
    melted_df,
    x="Input",
    y="Spine percentage",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

for n, row in dataframe.iterrows():
    perc_CA3_apical_plot.ax.plot(
        [0.3, 1.3],
        [row["% spines apical CA3"], row["% spines basal CA3"]],
        color='lightgray',
        zorder=0
    )

p_value = pairwise_comparison(
    melted_df[melted_df['Input'] == '% spines apical CA3']['Spine percentage'],
    melted_df[melted_df['Input'] == '% spines basal CA3']['Spine percentage'])

grouped_data = melted_df.groupby('Input').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Input', values='Spine percentage')

perc_CA3_apical_plot.annotate_stats(
    grouped_data,
    pairs=[("% spines apical CA3", "% spines basal CA3")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[0],
    draw_feet=True,
    feet_length=0.2,
    height_offset=2,
    text_offset=2/100,)

plt.tight_layout()

# perc_CA3_apical_plot.save_figure(
#     saving_path / 'perc_CA3_apical_plot.png',
#     dpi=dpi)

# %% 20) % of "activated" spines BLA per branch deg. apical vs basal dendrites

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch apical BLA'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_apical_df = pd.DataFrame(
    dataframe['% spines per branch apical BLA'].tolist(),
    columns=columns
)
spines_per_branch_basal_df = pd.DataFrame(
    dataframe['% spines per branch basal BLA'].tolist(),
    columns=columns
)

apical_long = spines_per_branch_apical_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
apical_long["Compartment"] = "Apical"

basal_long = spines_per_branch_basal_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
basal_long["Compartment"] = "Basal"

combined_df = pd.concat([apical_long, basal_long]).reset_index(drop=True)
combined_df.dropna(inplace=True)
combined_df['Group'] = (
    combined_df['Branch_Degree'].astype(str) + "_" +
    combined_df['Compartment'])


perc_spines_per_branch_apical_vs_basal_BLA_plot = DataPlotter(
    figsize=(5, 3),
    font_size=fontsize,
    y_lims=(-1, 15),
    y_label=r'% BLA spines w/ Ca$^{+2}$ event',
    x_tick_labels=([str(i + 1) for i in range(max_degree)]*2),
    x_ticks=(np.arange(8))
)

perc_spines_per_branch_apical_vs_basal_BLA_plot.box_plot(
    combined_df,
    x="Group",
    y="Spine_Density",
    box_width=0.05,
    color=BLA_color)

perc_spines_per_branch_apical_vs_basal_BLA_plot.strip_plot(
    combined_df,
    x="Group",
    y="Spine_Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.4)

perc_spines_per_branch_apical_vs_basal_BLA_plot.display_mean(
    combined_df,
    x="Group",
    y="Spine_Density",
    offset=0.4,
    width=0.1,
    linewidth=3,
    color='black',
    order=['1_Apical', '2_Apical', '3_Apical', '4_Apical',
           '1_Basal', '2_Basal', '3_Basal', '4_Basal'])

perc_spines_per_branch_apical_vs_basal_BLA_plot.ax.plot(
    [0, 3],
    [-3.5, -3.5],
    color='black',
    clip_on=False,
    lw=0.75)

perc_spines_per_branch_apical_vs_basal_BLA_plot.ax.plot(
    [4, 7],
    [-3.5, -3.52],
    color='black',
    clip_on=False,
    lw=0.75)

perc_spines_per_branch_apical_vs_basal_BLA_plot.ax.text(
    y=-5, x=1.5,
    s="Apical",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

perc_spines_per_branch_apical_vs_basal_BLA_plot.ax.text(
    y=-5, x=5.5,
    s="Basal",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

model = ols(
    'Spine_Density ~ C(Branch_Degree) * C(Compartment)',
    data=combined_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)


tukey = pairwise_tukeyhsd(
    endog=combined_df['Spine_Density'],
    groups=combined_df['Group'],  # use the combined group label
    alpha=0.05
)

grouped_data = combined_df.groupby('Group').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Group', values='Spine_Density')

perc_spines_per_branch_apical_vs_basal_BLA_plot.annotate_stats(
    grouped_data,
    pairs=[("3_Apical", "3_Basal"),
           ("3_Apical", "4_Basal")],
    p_values=[0.0037, 0.0196],
    x_positions=[(2.2, 6.2)],
    offsets=[2],
    draw_feet=True,
    feet_length=0.5,
    height_offset=2,
    text_offset=0.2,)

plt.tight_layout()

# perc_spines_per_branch_apical_vs_basal_BLA_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_apical_vs_basal_BLA.png',
#     dpi=dpi)

# %% 21) % of "activated" spines CA3 per branch deg. apical vs basal dendrites

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch apical CA3'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_apical_df = pd.DataFrame(
    dataframe['% spines per branch apical CA3'].tolist(),
    columns=columns
)
spines_per_branch_basal_df = pd.DataFrame(
    dataframe['% spines per branch basal CA3'].tolist(),
    columns=columns
)

apical_long = spines_per_branch_apical_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
apical_long["Compartment"] = "Apical"

basal_long = spines_per_branch_basal_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
basal_long["Compartment"] = "Basal"

combined_df = pd.concat([apical_long, basal_long]).reset_index(drop=True)
combined_df.dropna(inplace=True)
combined_df['Group'] = (
    combined_df['Branch_Degree'].astype(str) + "_" +
    combined_df['Compartment'])


perc_spines_per_branch_apical_vs_basal_CA3_plot = DataPlotter(
    figsize=(5, 3),
    font_size=fontsize,
    y_lims=(-1, 15),
    y_label=r'% CA3 spines w/ Ca$^{+2}$ event',
    x_tick_labels=([str(i + 1) for i in range(max_degree)]*2),
    x_ticks=(np.arange(8))
)

perc_spines_per_branch_apical_vs_basal_CA3_plot.box_plot(
    combined_df,
    x="Group",
    y="Spine_Density",
    box_width=0.05,
    color=CA3_color)

perc_spines_per_branch_apical_vs_basal_CA3_plot.strip_plot(
    combined_df,
    x="Group",
    y="Spine_Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.4)

perc_spines_per_branch_apical_vs_basal_CA3_plot.display_mean(
    combined_df,
    x="Group",
    y="Spine_Density",
    offset=0.4,
    width=0.1,
    linewidth=3,
    color='black',
    order=['1_Apical', '2_Apical', '3_Apical', '4_Apical',
           '1_Basal', '2_Basal', '3_Basal', '4_Basal'])

perc_spines_per_branch_apical_vs_basal_CA3_plot.ax.plot(
    [0, 3],
    [-3.5, -3.5],
    color='black',
    clip_on=False,
    lw=0.75)

perc_spines_per_branch_apical_vs_basal_CA3_plot.ax.plot(
    [4, 7],
    [-3.5, -3.52],
    color='black',
    clip_on=False,
    lw=0.75)

perc_spines_per_branch_apical_vs_basal_CA3_plot.ax.text(
    y=-5, x=1.5,
    s="Apical",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

perc_spines_per_branch_apical_vs_basal_CA3_plot.ax.text(
    y=-5, x=5.5,
    s="Basal",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

model = ols(
    'Spine_Density ~ C(Branch_Degree) * C(Compartment)',
    data=combined_df).fit()
anova_table = sm.stats.anova_lm(model, typ=2)


tukey = pairwise_tukeyhsd(
    endog=combined_df['Spine_Density'],
    groups=combined_df['Group'],  # use the combined group label
    alpha=0.05
)

grouped_data = combined_df.groupby('Group').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Group', values='Spine_Density')

plt.tight_layout()

# perc_spines_per_branch_apical_vs_basal_CA3_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_apical_vs_basal_CA3.png',
#     dpi=dpi)

# %% 22) % of "activated" spines BLA or CA3 per branch degree

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch degree BLA'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_degrees_df = pd.DataFrame(
    dataframe['% spines per branch degree BLA'].tolist(),
    columns=columns
)

melted_df = spines_per_branch_degrees_df.melt(
    var_name="Branch degree",
    value_name="Value")

perc_spines_per_branch_degree_BLA_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 15),
    y_label=r'% BLA spines w/ Ca$^{+2}$ event',
)

perc_spines_per_branch_degree_BLA_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=BLA_color)

perc_spines_per_branch_degree_BLA_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_spines_per_branch_degree_BLA_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

perc_spines_per_branch_degree_BLA_plot.ax.text(
    y=-3, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row['% spines per branch degree BLA']
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    perc_spines_per_branch_degree_BLA_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

p_value, multiple_comparison = group_comparison(
    spines_per_branch_degrees_df[[1, 2, 3]])

grouped_data = melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Branch degree', values='Value')

plt.tight_layout()

# perc_spines_per_branch_degree_BLA_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_degree_BLA.png',
#     dpi=dpi)

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch degree CA3'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_degrees_df = pd.DataFrame(
    dataframe['% spines per branch degree CA3'].tolist(),
    columns=columns
)

melted_df = spines_per_branch_degrees_df.melt(
    var_name="Branch degree",
    value_name="Value")

perc_spines_per_branch_degree_CA3_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 15),
    y_label=r'% CA3 spines w/ Ca$^{+2}$ event',
)

perc_spines_per_branch_degree_CA3_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=CA3_color)

perc_spines_per_branch_degree_CA3_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_spines_per_branch_degree_CA3_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

perc_spines_per_branch_degree_CA3_plot.ax.text(
    y=-3, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row['% spines per branch degree CA3']
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    perc_spines_per_branch_degree_CA3_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

p_value, multiple_comparison = group_comparison(
    spines_per_branch_degrees_df[[1, 2, 3]])

plt.tight_layout()

# perc_spines_per_branch_degree_CA3_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_degree_CA3.png',
#     dpi=dpi)

# %% 23) % of "activated" spines BLA or CA3 per branch degree
#        in apical dendrites

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch apical BLA'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_apical_df = pd.DataFrame(
    dataframe['% spines per branch apical BLA'].tolist(),
    columns=columns
)

melted_df = spines_per_branch_apical_df.melt(
    var_name="Branch degree",
    value_name="Value")

perc_spines_per_branch_apical_BLA_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 15),
    y_label=r'% BLA spines w/ Ca$^{+2}$ event',
)

perc_spines_per_branch_apical_BLA_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=BLA_color)

perc_spines_per_branch_apical_BLA_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_spines_per_branch_apical_BLA_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

perc_spines_per_branch_apical_BLA_plot.ax.text(
    y=-3, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row['% spines per branch apical BLA']
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    perc_spines_per_branch_apical_BLA_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

p_value, multiple_comparison = group_comparison(
    spines_per_branch_apical_df[[1, 2, 3]])

grouped_data = melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Branch degree', values='Value')

plt.tight_layout()

# perc_spines_per_branch_apical_BLA_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_apical_BLA.png',
#     dpi=dpi)

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch apical CA3'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_apical_df = pd.DataFrame(
    dataframe['% spines per branch apical CA3'].tolist(),
    columns=columns
)

melted_df = spines_per_branch_apical_df.melt(
    var_name="Branch degree",
    value_name="Value")

perc_spines_per_branch_apical_CA3_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 15),
    y_label=r'% CA3 spines w/ Ca$^{+2}$ event',
)

perc_spines_per_branch_apical_CA3_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=CA3_color)

perc_spines_per_branch_apical_CA3_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_spines_per_branch_apical_CA3_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

perc_spines_per_branch_apical_CA3_plot.ax.text(
    y=-3, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row['% spines per branch apical CA3']
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    perc_spines_per_branch_apical_CA3_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

p_value, multiple_comparison = group_comparison(
    spines_per_branch_apical_df[[1, 2, 3]])

plt.tight_layout()

# perc_spines_per_branch_apical_CA3_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_apical_CA3.png',
#     dpi=dpi)

# %% 24) % of "activated" spines BLA or CA3 per branch degree
#        in basal dendrites

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch basal BLA'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_basal_df = pd.DataFrame(
    dataframe['% spines per branch basal BLA'].tolist(),
    columns=columns
)

melted_df = spines_per_branch_basal_df.melt(
    var_name="Branch degree",
    value_name="Value")

perc_spines_per_branch_basal_BLA_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(-1, 15),
    y_label=r'% BLA spines w/ Ca$^{+2}$ event',
)

perc_spines_per_branch_basal_BLA_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=BLA_color)

perc_spines_per_branch_basal_BLA_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_spines_per_branch_basal_BLA_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

perc_spines_per_branch_basal_BLA_plot.ax.text(
    y=-3, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row['% spines per branch basal BLA']
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    perc_spines_per_branch_basal_BLA_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

p_value, multiple_comparison = group_comparison(
    spines_per_branch_basal_df[[1, 2, 3]])

grouped_data = melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Branch degree', values='Value')

plt.tight_layout()

# perc_spines_per_branch_basal_BLA_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_basal_BLA.png',
#     dpi=dpi)

max_degree = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch basal CA3'].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

spines_per_branch_basal_df = pd.DataFrame(
    dataframe['% spines per branch basal CA3'].tolist(),
    columns=columns
)

melted_df = spines_per_branch_basal_df.melt(
    var_name="Branch degree",
    value_name="Value")

perc_spines_per_branch_basal_CA3_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(-1, 15),
    y_label=r'% CA3 spines w/ Ca$^{+2}$ event',
)

perc_spines_per_branch_basal_CA3_plot.box_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=CA3_color)

perc_spines_per_branch_basal_CA3_plot.strip_plot(
    melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

perc_spines_per_branch_basal_CA3_plot.display_mean(
    melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

perc_spines_per_branch_basal_CA3_plot.ax.text(
    y=-3, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in dataframe.iterrows():

    y_values = [
        v for v in row['% spines per branch basal CA3']
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    perc_spines_per_branch_basal_CA3_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

p_value, multiple_comparison = group_comparison(
    spines_per_branch_basal_df[[1, 2, 3]])

plt.tight_layout()

# perc_spines_per_branch_basal_CA3_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_basal_CA3.png',
#     dpi=dpi)

# %% 25) % of "activated" spines BLA vs. CA3 per branch degree

# Prepare DataFrame for apical compartment
max_degree_apical = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch apical BLA'].tolist()])

columns_apical = np.arange(1, max_degree_apical+1, dtype=int)

spines_per_branch_apical_bla_df = pd.DataFrame(
    dataframe['% spines per branch apical BLA'].tolist(),
    columns=columns_apical)
spines_per_branch_apical_ca3_df = pd.DataFrame(
    dataframe['% spines per branch apical CA3'].tolist(),
    columns=columns_apical)

apical_bla_long = spines_per_branch_apical_bla_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
apical_bla_long["Input"] = "BLA"
apical_ca3_long = spines_per_branch_apical_ca3_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
apical_ca3_long["Input"] = "CA3"

apical_combined = pd.concat(
    [apical_bla_long, apical_ca3_long]).reset_index(drop=True)
apical_combined.dropna(inplace=True)

perc_spines_per_branch_apical_BLA_CA3_plot = DataPlotter(
    figsize=(5, 3),
    font_size=fontsize,
    y_lims=(-1, 15),
    y_label=r'% spines w/ Ca$^{+2}$ event',
    x_label="Branch degree"
)

sns.boxplot(
    apical_combined,
    x="Branch_Degree",
    y="Spine_Density",
    hue="Input",
    palette={"BLA": BLA_color, "CA3": CA3_color},
    ax=perc_spines_per_branch_apical_BLA_CA3_plot.ax,
    linecolor='black',
    linewidth=1,
    width=0.8,
    gap=0.4,
    legend=False,
    fliersize=0
    )

sns.stripplot(
    apical_combined,
    x="Branch_Degree",
    y="Spine_Density",
    hue="Input",
    dodge=True,
    edgecolor=gray_color,
    palette={"BLA": 'white', "CA3": 'white'},
    linewidth=1.5,
    marker='o',
    jitter=0,
    color=None,
    ax=perc_spines_per_branch_apical_BLA_CA3_plot.ax,
    legend=False,
    zorder=0,
)
strip_data = {
    "x": [],
    "y": []
}
for collection in perc_spines_per_branch_apical_BLA_CA3_plot.ax.collections:
    if hasattr(collection, 'get_offsets'):
        offsets = collection.get_offsets()
        if offsets is not None:

            offsets[:, 0] += 0.2
            collection.set_offsets(offsets)
            strip_data["x"].extend(offsets[:, 0])
            strip_data["y"].extend(offsets[:, 1])

strip_df = pd.DataFrame(strip_data, columns=["x", "y"])
mean_per_group = strip_df.groupby("x")["y"].mean()
sem_per_group = strip_df.groupby("x")["y"].sem()

for x_pos, mean_val in mean_per_group.items():
    sem_val = sem_per_group[x_pos]

    perc_spines_per_branch_apical_BLA_CA3_plot.ax.hlines(
        y=mean_val, xmin=x_pos - 0.06, xmax=x_pos + 0.06,
        colors='black', linewidth=3, zorder=2
    )
    perc_spines_per_branch_apical_BLA_CA3_plot.ax.vlines(
        x=x_pos, ymin=mean_val - sem_val, ymax=mean_val + sem_val,
        colors='black', linewidth=3, zorder=2
    )

plt.tight_layout()

# perc_spines_per_branch_apical_BLA_CA3_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_apical_BLA_CA3.png',
#     dpi=dpi)

# Run two-way ANOVA for apical compartment
model_apical = ols(
    'Spine_Density ~ C(Branch_Degree) * C(Input)', data=apical_combined).fit()
anova_table_apical = sm.stats.anova_lm(model_apical, typ=2)
print("ANOVA Table for Apical Compartment:")
print(anova_table_apical)

# --- Basal Compartment Analysis ---

max_degree_basal = np.max(
    [len(entry)
     for entry in dataframe['% spines per branch basal BLA'].tolist()])
columns_basal = np.arange(1, max_degree_basal+1, dtype=int)

spines_per_branch_basal_bla_df = pd.DataFrame(
    dataframe['% spines per branch basal BLA'].tolist(),
    columns=columns_basal)
spines_per_branch_basal_ca3_df = pd.DataFrame(
    dataframe['% spines per branch basal CA3'].tolist(),
    columns=columns_basal)

basal_bla_long = spines_per_branch_basal_bla_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
basal_bla_long["Input"] = "BLA"
basal_ca3_long = spines_per_branch_basal_ca3_df.melt(
    ignore_index=False, var_name="Branch_Degree", value_name="Spine_Density")
basal_ca3_long["Input"] = "CA3"

basal_combined = pd.concat(
    [basal_bla_long, basal_ca3_long]).reset_index(drop=True)
basal_combined.dropna(inplace=True)

perc_spines_per_branch_basal_BLA_CA3_plot = DataPlotter(
    figsize=(5, 3),
    font_size=fontsize,
    y_lims=(-1, 15),
    y_label=r'% spines w/ Ca$^{+2}$ event',
)

sns.boxplot(
    basal_combined,
    x="Branch_Degree",
    y="Spine_Density",
    hue="Input",
    palette={"BLA": BLA_color, "CA3": CA3_color},
    ax=perc_spines_per_branch_basal_BLA_CA3_plot.ax,
    linecolor='black',
    linewidth=1,
    width=0.8,
    gap=0.4,
    legend=False,
    fliersize=0
    )

sns.stripplot(
    basal_combined,
    x="Branch_Degree",
    y="Spine_Density",
    hue="Input",
    dodge=True,
    edgecolor=gray_color,
    palette={"BLA": 'white', "CA3": 'white'},
    linewidth=1.5,
    marker='o',
    jitter=0,
    color=None,
    ax=perc_spines_per_branch_basal_BLA_CA3_plot.ax,
    legend=False,
    zorder=0,
)
strip_data = {
    "x": [],
    "y": []
}
for collection in perc_spines_per_branch_basal_BLA_CA3_plot.ax.collections:
    if hasattr(collection, 'get_offsets'):
        offsets = collection.get_offsets()
        if offsets is not None:

            offsets[:, 0] += 0.2
            collection.set_offsets(offsets)
            strip_data["x"].extend(offsets[:, 0])
            strip_data["y"].extend(offsets[:, 1])

strip_df = pd.DataFrame(strip_data, columns=["x", "y"])
mean_per_group = strip_df.groupby("x")["y"].mean()
sem_per_group = strip_df.groupby("x")["y"].sem()

for x_pos, mean_val in mean_per_group.items():
    sem_val = sem_per_group[x_pos]

    perc_spines_per_branch_basal_BLA_CA3_plot.ax.hlines(
        y=mean_val, xmin=x_pos - 0.06, xmax=x_pos + 0.06,
        colors='black', linewidth=3, zorder=2
    )
    perc_spines_per_branch_basal_BLA_CA3_plot.ax.vlines(
        x=x_pos, ymin=mean_val - sem_val, ymax=mean_val + sem_val,
        colors='black', linewidth=3, zorder=2
    )

plt.tight_layout()

# perc_spines_per_branch_basal_BLA_CA3_plot.save_figure(
#     saving_path / 'perc_spines_per_branch_basal_BLA_CA3.png',
#     dpi=dpi)

model_basal = ols(
    'Spine_Density ~ C(Branch_Degree) * C(Input)', data=basal_combined).fit()
anova_table_basal = sm.stats.anova_lm(model_basal, typ=2)
print("ANOVA Table for Basal Compartment:")
print(anova_table_basal)

# %% 26) % "activated" spines branch-by-branch over total N of spines

for n, (segmenter, neurites) in enumerate(
        zip(segmenter_list, spynalyzer_list)):

    is_event_BLA = segmenter_list[n].calcium_events_BLA
    is_event_CA3 = segmenter_list[n].calcium_events_CA3

    probabilities_BLA = np.array(
        [sweep for spine in is_event_BLA for sweep in spine]
    )
    probabilities_BLA = probabilities_BLA[~np.isnan(probabilities_BLA)]
    dynamic_threshold_BLA = np.percentile(probabilities_BLA, 99)

    probabilities_CA3 = np.array(
        [sweep for spine in is_event_CA3 for sweep in spine]
    )
    probabilities_CA3 = probabilities_CA3[~np.isnan(probabilities_CA3)]
    dynamic_threshold_CA3 = np.percentile(probabilities_CA3, 99)

    fig, ax = plt.subplots(
        figsize=(7, 2.5),
        layout='constrained')

    for j, neurite in enumerate(neurites):

        if neurite:
            binary_event_BLA = np.zeros_like(neurite.calcium_events_BLA)
            binary_event_CA3 = np.zeros_like(neurite.calcium_events_CA3)

            for i, (spine_BLA, spine_CA3) in enumerate(zip(
                    neurite.calcium_events_BLA,
                    neurite.calcium_events_CA3)):

                for k, (sweep_BLA, sweep_CA3) in enumerate(zip(
                        spine_BLA, spine_CA3)):

                    if sweep_BLA > dynamic_threshold_BLA:
                        binary_event_BLA[i][k] = 1
                    if sweep_CA3 > dynamic_threshold_CA3:
                        binary_event_CA3[i][k] = 1

            BLA_count = 0
            for spine in binary_event_BLA:
                if sum(spine) > 0:
                    BLA_count += 1

            CA3_count = 0
            for spine in binary_event_CA3:
                if sum(spine) > 0:
                    CA3_count += 1

            try:
                BLA_perc = (BLA_count / neurite.n_spines) * 100
            except ZeroDivisionError:
                BLA_perc = 0
            try:
                CA3_perc = (CA3_count / neurite.n_spines) * 100
            except ZeroDivisionError:
                CA3_perc = 0

            ax.scatter(
                j,
                BLA_perc,
                edgecolor=BLA_color,
                facecolor='none',
                linewidth=1.5,
                clip_on=False)
            ax.scatter(
                j,
                CA3_perc,
                edgecolor=CA3_color,
                facecolor='none',
                linewidth=1.5,
                clip_on=False)

            ax.vlines(
                j,
                ymin=BLA_perc,
                ymax=CA3_perc,
                color=(
                    'red' if neurite.compartment == 'apical dendrite'
                    else 'blue'),
                alpha=0.3,
                zorder=0)
            ax.axhline(
                0,
                color='lightgray',
                ls='dashed',
                zorder=0)

    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_position(('outward', 20))
    ax.set_ylim(0, 20)
    ax.set_yticks([0, 10, 20])
    ax.set_xlim(0, neurites.n_neurites)
    ax.set_xticks(np.arange(0, neurites.n_neurites, 5))
    ax.set_ylabel('% spine w/' + '\n' + r'Ca$^{+2}$ event', fontsize=fontsize)
    ax.set_xlabel('Branch #', fontsize=fontsize)
    ax.tick_params(labelsize=fontsize)

# %% 27) % ratio of "activated" spines branch-by-branch
#        over total "activated" spines

BLA_perc_list_all_neurons = []
CA3_perc_list_all_neurons = []

for n, (segmenter, neurites) in enumerate(
        zip(segmenter_list, spynalyzer_list)):

    path = neurites.dataset.folder
    name = f"{path.parts[-3]}_{path.parts[-2]}"

    bla_total = sum(
        [1 if sum(spine) > 0 else 0
         for spine in segmenter.calcium_events_binary_BLA])
    ca3_total = sum(
        [1 if sum(spine) > 0 else 0
         for spine in segmenter.calcium_events_binary_CA3])

    non_none_neurites = [
        neurite for neurite in neurites if neurite is not None]
    # sorted_neurites = sorted(
    #     non_none_neurites, key=lambda neurite: neurite.n_spines)

    fig, ax = plt.subplots(
        2, 1,
        figsize=(7, 5),
        layout='constrained',
        sharex=True,
        gridspec_kw={
            'hspace': 0.2,
            'height_ratios': (1, 2)})

    BLA_perc_list = []
    CA3_perc_list = []

    neurite_count = 0
    for j, neurite in enumerate(non_none_neurites):

        if neurite.spines:
            try:
                BLA_perc = (neurite.n_spines_BLA / bla_total)
            except ZeroDivisionError:
                BLA_perc = 0
            try:
                CA3_perc = (neurite.n_spines_CA3 / ca3_total)
            except ZeroDivisionError:
                CA3_perc = 0

            ax[1].scatter(
                neurite_count,
                BLA_perc,
                edgecolor=BLA_color,
                facecolor='none',
                linewidth=1.5,
                clip_on=False)
            ax[1].scatter(
                neurite_count,
                CA3_perc,
                edgecolor=CA3_color,
                facecolor='none',
                linewidth=1.5,
                clip_on=False)

            ax[1].vlines(
                neurite_count,
                ymin=BLA_perc,
                ymax=CA3_perc,
                color=(
                    'red' if neurite.compartment == 'apical dendrite'
                    else 'blue'),
                alpha=0.3,
                zorder=0,
                clip_on=False)
            ax[1].axhline(
                0,
                color='lightgray',
                ls='dashed',
                zorder=0)

            BLA_perc_list.append(BLA_perc)
            CA3_perc_list.append(CA3_perc)

            BLA_perc_dict = {name: BLA_perc_list}
            CA3_perc_dict = {name: CA3_perc_list}

            neurite_count += 1

    BLA_perc_list_all_neurons.append(BLA_perc_dict)
    CA3_perc_list_all_neurons.append(CA3_perc_dict)

    domination_coefficient = [
        bla - ca3 for bla, ca3 in zip(BLA_perc_list, CA3_perc_list)]

    ax[0].plot(
        np.arange(sum(
            [1 if neurite.spines else 0
             for neurite in non_none_neurites])),
        domination_coefficient,
        color='black',
        lw=1.5)
    ax[0].axhline(
        0,
        color='lightgray',
        ls='dashed',
        zorder=0)

    ax[0].spines[['top', 'right', 'bottom']].set_visible(False)
    ax[0].spines['left'].set_position(('outward', 20))
    ax[0].set_ylim(-0.4, 0.4)
    ax[0].set_yticks([-0.4, 0, 0.4])
    ax[0].tick_params(axis='x', length=0)
    ax[0].tick_params(axis='y', labelsize=fontsize)
    ax[0].set_xticks([])
    ax[0].set_ylabel('Domination\ncoefficient (%)', fontsize=fontsize)

    ax[1].spines[['top', 'right']].set_visible(False)
    ax[1].spines[['bottom', 'left']].set_position(('outward', 20))
    ax[1].set_ylim(0, 0.4)
    ax[1].set_yticks([0, 0.2, 0.4])
    ax[1].set_xlim(0, neurites.n_neurites)
    ax[1].set_xticks(np.arange(0, neurites.n_neurites, 5))
    ax[1].set_ylabel('Ratio of spine w/' + '\n' + r'Ca$^{+2}$ event',
                     fontsize=fontsize)
    ax[1].set_xlabel('Branch #', fontsize=fontsize)
    ax[1].tick_params(axis='both', labelsize=fontsize)

    # fig.savefig(
    #     saving_path / f'{name}_active_spines_ratio_per_branch.png',
    #     dpi=dpi)


# %% 28) N of "activated" spines per compartment

BLA_spines_per_compartment_df = pd.DataFrame({
        'Apical': dataframe["N spines per apical BLA"].tolist(),
        'Basal': dataframe["N spines per basal BLA"].tolist()
    })

BLA_spines_per_compartment_melted_df = BLA_spines_per_compartment_df.melt(
    var_name="Compartment",
    value_name="N spines")


BLA_spines_per_compartment_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='N of spines',
)

BLA_spines_per_compartment_plot.box_plot(
    BLA_spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    box_width=0.15,
    color=BLA_color)

BLA_spines_per_compartment_plot.strip_plot(
    BLA_spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

BLA_spines_per_compartment_plot.display_mean(
    BLA_spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

x_values = [0.3, 1.3]

for n, row in BLA_spines_per_compartment_df.iterrows():

    y_values = [
        v for v in row
        if not pd.isna(v)]

    BLA_spines_per_compartment_plot.ax.plot(
        x_values,
        y_values,
        color='lightgray',
        zorder=0)

p_value = pairwise_comparison(
    dataframe["N spines per apical BLA"].to_numpy(),
    dataframe["N spines per basal BLA"].to_numpy()
    )

grouped_data = BLA_spines_per_compartment_melted_df.groupby(
    'Compartment').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Compartment', values='N spines')

BLA_spines_per_compartment_plot.annotate_stats(
    grouped_data,
    pairs=[("Apical", "Basal")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[10],
    draw_feet=True,
    feet_length=2,
    height_offset=2,
    text_offset=0.2,)

plt.tight_layout()

# BLA_spines_per_compartment_plot.save_figure(
#     saving_path / 'BLA_spines_per_compartment.png',
#     dpi=dpi)

CA3_spines_per_compartment_df = pd.DataFrame({
        'Apical': dataframe["N spines per apical CA3"].tolist(),
        'Basal': dataframe["N spines per basal CA3"].tolist()
    })

CA3_spines_per_compartment_melted_df = CA3_spines_per_compartment_df.melt(
    var_name="Compartment",
    value_name="N spines")


CA3_spines_per_compartment_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='N of spines',
)

CA3_spines_per_compartment_plot.box_plot(
    CA3_spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    box_width=0.15,
    color=CA3_color)

CA3_spines_per_compartment_plot.strip_plot(
    CA3_spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

CA3_spines_per_compartment_plot.display_mean(
    CA3_spines_per_compartment_melted_df,
    x="Compartment",
    y="N spines",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

x_values = [0.3, 1.3]

for n, row in CA3_spines_per_compartment_df.iterrows():

    y_values = [
        v for v in row
        if not pd.isna(v)]

    CA3_spines_per_compartment_plot.ax.plot(
        x_values,
        y_values,
        color='lightgray',
        zorder=0)

p_value = pairwise_comparison(
    dataframe["N spines per apical CA3"].to_numpy(),
    dataframe["N spines per basal CA3"].to_numpy()
    )

grouped_data = CA3_spines_per_compartment_melted_df.groupby(
    'Compartment').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Compartment', values='N spines')

CA3_spines_per_compartment_plot.annotate_stats(
    grouped_data,
    pairs=[("Apical", "Basal")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[10],
    draw_feet=True,
    feet_length=2,
    height_offset=2,
    text_offset=0.2,)

plt.tight_layout()

# CA3_spines_per_compartment_plot.save_figure(
#     saving_path / 'CA3_spines_per_compartment.png',
#     dpi=dpi)


# %% 29) N of "activated" spines per branch type

max_degree = np.max(
    [len(entry)
     for entry in dataframe["N spines per branch degree BLA"].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

BLA_per_branch_degrees_df = pd.DataFrame(
    dataframe["N spines per branch degree BLA"].tolist(),
    columns=columns
)

BLA_melted_df = BLA_per_branch_degrees_df.melt(
    var_name="Branch degree",
    value_name="Value")

BLA_branches_order_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 60),
    y_label='N of BLA spines',
)

BLA_branches_order_plot.box_plot(
    BLA_melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=BLA_color)

BLA_branches_order_plot.strip_plot(
    BLA_melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

BLA_branches_order_plot.display_mean(
    BLA_melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

BLA_branches_order_plot.ax.text(
    y=-12, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in BLA_per_branch_degrees_df.iterrows():

    y_values = [
        v for v in dataframe["N spines per branch degree BLA"].iloc[n]
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    BLA_branches_order_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

plt.tight_layout()

# BLA_branches_order_plot.save_figure(
#     saving_path / 'n_spines_per_branch_BLA_order.png',
#     dpi=dpi)

max_degree = np.max(
    [len(entry)
     for entry in dataframe["N spines per branch degree CA3"].tolist()])
columns = np.arange(1, max_degree+1, dtype=int)

CA3_per_branch_degrees_df = pd.DataFrame(
    dataframe["N spines per branch degree CA3"].tolist(),
    columns=columns
)

CA3_melted_df = CA3_per_branch_degrees_df.melt(
    var_name="Branch degree",
    value_name="Value")

CA3_branches_order_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 60),
    y_label='N of CA3 spines',
)

CA3_branches_order_plot.box_plot(
    CA3_melted_df,
    x="Branch degree",
    y="Value",
    box_width=0.1,
    color=CA3_color)

CA3_branches_order_plot.strip_plot(
    CA3_melted_df,
    x="Branch degree",
    y="Value",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

CA3_branches_order_plot.display_mean(
    CA3_melted_df,
    x="Branch degree",
    y="Value",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

CA3_branches_order_plot.ax.text(
    y=-12, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize)

x_values = np.arange(0.3, 4.3, 1)

for n, row in CA3_per_branch_degrees_df.iterrows():

    y_values = [
        v for v in dataframe["N spines per branch degree CA3"].iloc[n]
        if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    CA3_branches_order_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0)

plt.tight_layout()

# CA3_branches_order_plot.save_figure(
#     saving_path / 'n_spines_per_branch_CA3_order.png',
#     dpi=dpi)

# %% 30) Density of spines per compartment

compartment_length = pd.DataFrame({
    "Dendritic length": [list(lengths) for lengths in zip(
        dataframe["Dendritic length apical"],
        dataframe["Dendritic length basal"]
    )]
})

density_per_compartment = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        [
            [apical, basal]
            for apical, basal in zip(
                dataframe["N spines per apical"].to_list(),
                dataframe["N spines per basal"].to_list()
            )
        ],
        compartment_length["Dendritic length"].to_list())
]


spine_density_df = pd.DataFrame(
    density_per_compartment,
    columns=['Apical', 'Basal']
)

spine_density_melted_df = spine_density_df.melt(
    var_name="Compartment",
    value_name="Density"
)


density_per_compartment_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 1000),
    y_label='Spine density (spines/mm)',
    y_ticks=np.linspace(0, 1000, 5),
    y_tick_labels=np.linspace(0, 1000, 5, dtype=int)
)

density_per_compartment_plot.box_plot(
    spine_density_melted_df,
    x="Compartment",
    y="Density",
    box_width=0.15,
    color=gray_color)

density_per_compartment_plot.strip_plot(
    spine_density_melted_df,
    x="Compartment",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

density_per_compartment_plot.display_mean(
    spine_density_melted_df,
    x="Compartment",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

x_values = [0.3, 1.3]

for n, row in spine_density_df.iterrows():

    y_values = [
        v for v in row
        if not pd.isna(v)]

    density_per_compartment_plot.ax.plot(
        x_values,
        y_values,
        color='lightgray',
        zorder=0)

p_value = pairwise_comparison(
    spine_density_df['Apical'], spine_density_df['Basal'])

grouped_data = spine_density_melted_df.groupby(
    'Compartment').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Compartment', values='Density')

density_per_compartment_plot.annotate_stats(
    grouped_data,
    pairs=[("Apical", "Basal")],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[200],
    draw_feet=True,
    feet_length=20,
    height_offset=2,
    text_offset=2/100,)

plt.tight_layout()

# density_per_compartment_plot.save_figure(
#     saving_path / 'spine_density_per_compartment.png',
#     dpi=dpi)

# %% 31) Density of spines per branch degree

spine_density_per_branch = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch degree"],
        dataframe["Branch degree lengths"])
]

max_degree = np.max(
    [len(entry)
     for entry in spine_density_per_branch])
columns = np.arange(1, max_degree+1, dtype=int)


spine_density_df = pd.DataFrame(
    spine_density_per_branch,
    columns=columns
)

spine_density_melted_df = spine_density_df.melt(
    var_name="Branch degree",
    value_name="Density"
)
spine_density_melted_df[
    "Branch degree"] = spine_density_melted_df["Branch degree"]

spine_density_plot_per_branch_degree = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 1200),
    y_ticks=np.linspace(0, 1200, 4),
    y_tick_labels=np.linspace(0, 1200, 4, dtype=int),
    y_label='Spines density (spines/mm)',
)

spine_density_plot_per_branch_degree.box_plot(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=gray_color,
)

spine_density_plot_per_branch_degree.strip_plot(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

spine_density_plot_per_branch_degree.display_mean(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

spine_density_plot_per_branch_degree.ax.text(
    y=-200, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in spine_density_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    spine_density_plot_per_branch_degree.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

p_value, multiple_comparison = group_comparison(
    spine_density_df[[1, 2, 3]])

grouped_data = spine_density_melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Branch degree', values='Density')

spine_density_plot_per_branch_degree.annotate_stats(
    grouped_data,
    pairs=[(1, 2),
           (1, 3),
           (2, 3)],
    p_values=[0.075312, 0.128777, 1.],
    x_positions=[(0.15, 1.15),
                 (0.15, 2.15),
                 (1.15, 2.15)],
    offsets=[100, 380, 230],
    draw_feet=True,
    feet_length=20,
    height_offset=2,
    text_offset=2/100,)

plt.subplots_adjust(top=1., bottom=0.2)

# spine_density_plot_per_branch_degree.save_figure(
#     saving_path / 'spine_density_per_branch_degree.png',
#     dpi=dpi)


# %% 32) Density of spines per branch degree in apical dendrites

spine_density_per_branch_apical = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch apical"],
        dataframe["Branch degree lengths apical"])
]

max_degree = np.max(
    [len(entry)
     for entry in spine_density_per_branch_apical])
columns = np.arange(1, max_degree+1, dtype=int)

spine_density_df = pd.DataFrame(
    spine_density_per_branch_apical,
    columns=columns
)

spine_density_melted_df = spine_density_df.melt(
    var_name="Branch degree",
    value_name="Density"
)
spine_density_melted_df[
    "Branch degree"] = spine_density_melted_df["Branch degree"]

spine_density_plot_per_branch_degree_apical = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 1200),
    y_ticks=np.linspace(0, 1200, 4),
    y_tick_labels=np.linspace(0, 1200, 4, dtype=int),
    y_label='Spines density (spines/mm)',
)

spine_density_plot_per_branch_degree_apical.box_plot(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=gray_color,
)

spine_density_plot_per_branch_degree_apical.strip_plot(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

spine_density_plot_per_branch_degree_apical.display_mean(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

spine_density_plot_per_branch_degree_apical.ax.text(
    y=-200, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in spine_density_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    spine_density_plot_per_branch_degree_apical.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

p_value, multiple_comparison = group_comparison(
    spine_density_df[[1, 2, 3]])

grouped_data = spine_density_melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Branch degree', values='Density')

spine_density_plot_per_branch_degree_apical.annotate_stats(
    grouped_data,
    pairs=[(1, 2),
           (1, 3),
           (2, 3)],
    p_values=[0.14265, 0.01534, 1.00000],
    x_positions=[(0.15, 1.15),
                 (0.15, 2.15),
                 (1.15, 2.15)],
    offsets=[100, 380, 230],
    draw_feet=True,
    feet_length=20,
    height_offset=2,
    text_offset=2/100,)

plt.subplots_adjust(top=1., bottom=0.2)

# spine_density_plot_per_branch_degree_apical.save_figure(
#     saving_path / 'spine_density_per_branch_degree_apical.png',
#     dpi=dpi)

# %% 33) Density of spines per branch degree in basal dendrites

spine_density_per_branch_basal = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch basal"],
        dataframe["Branch degree lengths basal"])
]

max_degree = np.max([len(entry) for entry in spine_density_per_branch_basal])
columns = np.arange(1, max_degree+1)

spine_density_df = pd.DataFrame(
    spine_density_per_branch_basal,
    columns=columns
)

spine_density_melted_df = spine_density_df.melt(
    var_name="Branch degree",
    value_name="Density"
)
spine_density_melted_df[
    "Branch degree"] = spine_density_melted_df["Branch degree"]

spine_density_plot_per_branch_degree_basal = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 1200),
    y_ticks=np.linspace(0, 1200, 4),
    y_tick_labels=np.linspace(0, 1200, 4, dtype=int),
    y_label='Spines density (spines/mm)',
)

spine_density_plot_per_branch_degree_basal.box_plot(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=gray_color,
)

spine_density_plot_per_branch_degree_basal.strip_plot(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

spine_density_plot_per_branch_degree_basal.display_mean(
    spine_density_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

spine_density_plot_per_branch_degree_basal.ax.text(
    y=-200, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in spine_density_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    spine_density_plot_per_branch_degree_basal.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

p_value = pairwise_comparison(
    spine_density_df[1],
    spine_density_df[2])

grouped_data = spine_density_melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Branch degree', values='Density')

spine_density_plot_per_branch_degree_basal.annotate_stats(
    grouped_data,
    pairs=[(1, 2)],
    p_values=[p_value],
    x_positions=[(0.15, 1.15)],
    offsets=[100],
    draw_feet=True,
    feet_length=20,
    height_offset=2,
    text_offset=2/100,)

plt.subplots_adjust(top=1., bottom=0.2)

# spine_density_plot_per_branch_degree_basal.save_figure(
#     saving_path / 'spine_density_per_branch_degree_basal.png',
#     dpi=dpi)

# %% 34) Density of "activated" spines per compartment

compartment_length = pd.DataFrame({
    "Dendritic length": [list(lengths) for lengths in zip(
        dataframe["Dendritic length apical"],
        dataframe["Dendritic length basal"]
    )]
})

BLA_density_per_compartment = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        [
            [apical, basal]
            for apical, basal in zip(
                dataframe["N spines per apical BLA"].to_list(),
                dataframe["N spines per basal BLA"].to_list()
            )
        ],
        compartment_length["Dendritic length"].to_list())
]
BLA_spine_density_df = pd.DataFrame(
    BLA_density_per_compartment,
    columns=['Apical', 'Basal']
)

BLA_spine_density_melted_df = BLA_spine_density_df.melt(
    var_name="Compartment",
    value_name="Density"
)

BLA_density_per_compartment_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 50),
    y_label='BLA spine density (spines/mm)',
)

BLA_density_per_compartment_plot.box_plot(
    BLA_spine_density_melted_df,
    x="Compartment",
    y="Density",
    box_width=0.15,
    color=BLA_color)

BLA_density_per_compartment_plot.strip_plot(
    BLA_spine_density_melted_df,
    x="Compartment",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

BLA_density_per_compartment_plot.display_mean(
    BLA_spine_density_melted_df,
    x="Compartment",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

x_values = [0.3, 1.3]

for n, row in BLA_spine_density_df.iterrows():

    y_values = [
        v for v in row
        if not pd.isna(v)]

    BLA_density_per_compartment_plot.ax.plot(
        x_values,
        y_values,
        color='lightgray',
        zorder=0)

p_value_BLA = pairwise_comparison(
    BLA_spine_density_df['Apical'], BLA_spine_density_df['Basal'])

grouped_data = BLA_spine_density_melted_df.groupby(
    'Compartment').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Compartment', values='Density')

BLA_density_per_compartment_plot.annotate_stats(
    grouped_data,
    pairs=[("Apical", "Basal")],
    p_values=[p_value_BLA],
    x_positions=[(0.15, 1.15)],
    offsets=[10],
    draw_feet=True,
    feet_length=1,
    height_offset=2,
    text_offset=0.1,)

plt.tight_layout()

# BLA_density_per_compartment_plot.save_figure(
#     saving_path / 'BLA_density_per_compartment.png',
#     dpi=dpi)

CA3_density_per_compartment = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        [
            [apical, basal]
            for apical, basal in zip(
                dataframe["N spines per apical CA3"].to_list(),
                dataframe["N spines per basal CA3"].to_list()
            )
        ],
        compartment_length["Dendritic length"].to_list())
]
CA3_spine_density_df = pd.DataFrame(
    CA3_density_per_compartment,
    columns=['Apical', 'Basal']
)

CA3_spine_density_melted_df = CA3_spine_density_df.melt(
    var_name="Compartment",
    value_name="Density"
)

CA3_density_per_compartment_plot = DataPlotter(
    figsize=(2.5, 3),
    font_size=fontsize,
    y_lims=(0, 50),
    y_label='CA3 spine density (spines/mm)',
)

CA3_density_per_compartment_plot.box_plot(
    CA3_spine_density_melted_df,
    x="Compartment",
    y="Density",
    box_width=0.15,
    color=CA3_color)

CA3_density_per_compartment_plot.strip_plot(
    CA3_spine_density_melted_df,
    x="Compartment",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3)

CA3_density_per_compartment_plot.display_mean(
    CA3_spine_density_melted_df,
    x="Compartment",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black')

x_values = [0.3, 1.3]

for n, row in CA3_spine_density_df.iterrows():

    y_values = [
        v for v in row
        if not pd.isna(v)]

    CA3_density_per_compartment_plot.ax.plot(
        x_values,
        y_values,
        color='lightgray',
        zorder=0)

p_value_CA3 = pairwise_comparison(
    CA3_spine_density_df['Apical'], CA3_spine_density_df['Basal'])

grouped_data = CA3_spine_density_melted_df.groupby(
    'Compartment').max().reset_index()
grouped_data = grouped_data.pivot(
    columns='Compartment', values='Density')

CA3_density_per_compartment_plot.annotate_stats(
    grouped_data,
    pairs=[("Apical", "Basal")],
    p_values=[p_value_CA3],
    x_positions=[(0.15, 1.15)],
    offsets=[10],
    draw_feet=True,
    feet_length=1,
    height_offset=2,
    text_offset=0.1,)

plt.tight_layout()

# CA3_density_per_compartment_plot.save_figure(
#     saving_path / 'CA3_density_per_compartment.png',
#     dpi=dpi)


# %% 35) Density of "activated" spines per branch degree

BLA_density_per_branch = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch degree BLA"],
        dataframe["Branch degree lengths"])
]

max_degree = np.max(
    [len(entry)
     for entry in BLA_density_per_branch])
columns = np.arange(1, max_degree+1, dtype=int)

BLA_density_df = pd.DataFrame(
    BLA_density_per_branch,
    columns=columns
)

BLA_density_melted_df = BLA_density_df.melt(
    var_name="Branch degree",
    value_name="Density"
)
BLA_density_melted_df[
    "Branch degree"] = BLA_density_melted_df["Branch degree"]

BLA_density_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='BLA spines density (spines/mm)',
)

BLA_density_plot.box_plot(
    BLA_density_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=BLA_color,
)

BLA_density_plot.strip_plot(
    BLA_density_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

BLA_density_plot.display_mean(
    BLA_density_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

BLA_density_plot.ax.text(
    y=-12, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in BLA_density_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    BLA_density_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

p_value_BLA, multiple_comparison_BLA = group_comparison(
    BLA_density_df[[1, 2, 3]])

grouped_data_BLA = BLA_density_melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data_BLA = grouped_data_BLA.pivot(
    columns='Branch degree', values='Density')

BLA_density_plot.annotate_stats(
    grouped_data_BLA,
    pairs=[(1, 2),
           (1, 3),
           (2, 3)],
    p_values=[0.03744,
              0.03313,
              1.00000],
    x_positions=[(0.15, 1.15),
                 (0.15, 2.15),
                 (1.15, 2.15)],
    offsets=[10, 22, 8],
    draw_feet=True,
    feet_length=2,
    height_offset=2,
    text_offset=2/100,)

plt.subplots_adjust(top=1., bottom=0.2)

# BLA_density_plot.save_figure(
#     saving_path / 'BLA_density_per_branch_degree.png',
#     dpi=dpi)

CA3_density_per_branch = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch degree CA3"],
        dataframe["Branch degree lengths"])
]

max_degree = np.max(
    [len(entry)
     for entry in CA3_density_per_branch])
columns = np.arange(1, max_degree+1, dtype=int)

CA3_density_df = pd.DataFrame(
    CA3_density_per_branch,
    columns=columns
)

CA3_density_melted_df = CA3_density_df.melt(
    var_name="Branch degree",
    value_name="Density"
)
CA3_density_melted_df[
    "Branch degree"] = CA3_density_melted_df["Branch degree"].astype(int)

CA3_density_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='CA3 spines density (spines/mm)',
)

CA3_density_plot.box_plot(
    CA3_density_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=CA3_color,
)

CA3_density_plot.strip_plot(
    CA3_density_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

CA3_density_plot.display_mean(
    CA3_density_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

CA3_density_plot.ax.text(
    y=-12, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in CA3_density_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    CA3_density_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

p_value_CA3, multiple_comparison_CA3 = group_comparison(
    CA3_density_df[[1, 2, 3]])

plt.subplots_adjust(top=1., bottom=0.2)

# CA3_density_plot.save_figure(
#     saving_path / 'CA3_density_per_branch_degree.png',
#     dpi=dpi)


# %% 36) Density of "activated" spines per branch degree in apical dendrites

BLA_density_per_branch_apical = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch apical BLA"],
        dataframe["Branch degree lengths apical"])
]

max_degree = np.max(
    [len(entry)
     for entry in BLA_density_per_branch_apical])
columns = np.arange(1, max_degree+1, dtype=int)

BLA_density_apical_df = pd.DataFrame(
    BLA_density_per_branch_apical,
    columns=columns
)

BLA_density_apical_melted_df = BLA_density_apical_df.melt(
    var_name="Branch degree",
    value_name="Density"
)

BLA_density_per_branch_apical_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='BLA spines density (spines/mm)',
)

BLA_density_per_branch_apical_plot.box_plot(
    BLA_density_apical_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=BLA_color,
)

BLA_density_per_branch_apical_plot.strip_plot(
    BLA_density_apical_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

BLA_density_per_branch_apical_plot.display_mean(
    BLA_density_apical_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

BLA_density_per_branch_apical_plot.ax.text(
    y=-18, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in BLA_density_apical_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    BLA_density_per_branch_apical_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

p_value_BLA, multiple_comparison_BLA = group_comparison(
    BLA_density_apical_df[[1, 2, 3]])

grouped_data_BLA = BLA_density_apical_melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data_BLA = BLA_density_apical_melted_df.pivot(
    columns='Branch degree', values='Density')

plt.subplots_adjust(top=1., bottom=0.2)

# BLA_density_per_branch_apical_plot.save_figure(
#     saving_path / 'BLA_density_per_branch_apical.png',
#     dpi=dpi)

CA3_density_per_branch_apical = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch apical CA3"],
        dataframe["Branch degree lengths apical"])
]

max_degree = np.max(
    [len(entry)
     for entry in CA3_density_per_branch_apical])
columns = np.arange(1, max_degree+1, dtype=int)

CA3_density_apical_df = pd.DataFrame(
    CA3_density_per_branch_apical,
    columns=columns
)

CA3_density_apical_melted_df = CA3_density_apical_df.melt(
    var_name="Branch degree",
    value_name="Density"
)

CA3_density_per_branch_apical_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='CA3 spines density (spines/mm)',
)

CA3_density_per_branch_apical_plot.box_plot(
    CA3_density_apical_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=CA3_color,
)

CA3_density_per_branch_apical_plot.strip_plot(
    CA3_density_apical_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

CA3_density_per_branch_apical_plot.display_mean(
    CA3_density_apical_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

CA3_density_per_branch_apical_plot.ax.text(
    y=-18, x=1.5,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in CA3_density_apical_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    CA3_density_per_branch_apical_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

# p_value_CA3, multiple_comparison_CA3 = group_comparison(
#     CA3_density_apical_df[[1, 2, 3]])

plt.subplots_adjust(top=1., bottom=0.2)

# CA3_density_per_branch_apical_plot.save_figure(
#     saving_path / 'CA3_density_per_branch_apical.png',
#     dpi=dpi)

# %% 37) Density of "activated" spines per branch degree in basal dendrites

BLA_density_per_branch_basal = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch basal BLA"],
        dataframe["Branch degree lengths basal"])
]

max_degree = np.max(
    [len(entry)
     for entry in BLA_density_per_branch_basal])
columns = np.arange(1, max_degree+1, dtype=int)

BLA_density_basal_df = pd.DataFrame(
    BLA_density_per_branch_basal,
    columns=columns
)

BLA_density_basal_melted_df = BLA_density_basal_df.melt(
    var_name="Branch degree",
    value_name="Density"
)

BLA_density_per_branch_basal_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='BLA spines density (spines/mm)',
)

BLA_density_per_branch_basal_plot.box_plot(
    BLA_density_basal_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=BLA_color,
)

BLA_density_per_branch_basal_plot.strip_plot(
    BLA_density_basal_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

BLA_density_per_branch_basal_plot.display_mean(
    BLA_density_basal_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

BLA_density_per_branch_basal_plot.ax.text(
    y=-18, x=1.,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in BLA_density_basal_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    BLA_density_per_branch_basal_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

p_value_BLA = pairwise_comparison(
    BLA_density_basal_df[1],
    BLA_density_basal_df[2])

grouped_data_BLA = BLA_density_basal_melted_df.groupby(
    'Branch degree').max().reset_index()
grouped_data_BLA = BLA_density_basal_melted_df.pivot(
    columns='Branch degree', values='Density')

plt.subplots_adjust(top=1., bottom=0.2)

# BLA_density_per_branch_basal_plot.save_figure(
#     saving_path / 'BLA_density_per_branch_basal.png',
#     dpi=dpi)

CA3_density_per_branch_basal = [
    [spines / length if length > 0 else 0
     for spines, length in zip(spine_counts, length_counts)]
    for spine_counts, length_counts in zip(
        dataframe["N spines per branch basal CA3"],
        dataframe["Branch degree lengths basal"])
]

max_degree = np.max(
    [len(entry)
     for entry in CA3_density_per_branch_basal])
columns = np.arange(1, max_degree+1, dtype=int)

CA3_density_basal_df = pd.DataFrame(
    CA3_density_per_branch_basal,
    columns=[1, 2, 3]
)

CA3_density_basal_melted_df = CA3_density_basal_df.melt(
    var_name="Branch degree",
    value_name="Density"
)

CA3_density_per_branch_basal_plot = DataPlotter(
    figsize=(3.5, 3),
    font_size=fontsize,
    y_lims=(0, 80),
    y_label='CA3 spines density (spines/mm)',
)

CA3_density_per_branch_basal_plot.box_plot(
    CA3_density_basal_melted_df,
    x="Branch degree",
    y="Density",
    box_width=0.1,
    color=CA3_color,
)

CA3_density_per_branch_basal_plot.strip_plot(
    CA3_density_basal_melted_df,
    x="Branch degree",
    y="Density",
    linewidth=1.5,
    edgecolor=gray_color,
    fillcolor='none',
    jitter=0,
    strip_offset=0.3
)

CA3_density_per_branch_basal_plot.display_mean(
    CA3_density_basal_melted_df,
    x="Branch degree",
    y="Density",
    offset=0.3,
    width=0.1,
    linewidth=3,
    color='black'
)

CA3_density_per_branch_basal_plot.ax.text(
    y=-18, x=1.,
    s="Branch degree",
    verticalalignment='center',
    horizontalalignment='center',
    fontsize=fontsize
)

x_values = np.arange(0.3, 4.3, 1)

for n, row in CA3_density_basal_df.iterrows():
    y_values = [v for v in row if not pd.isna(v)]
    x_values_adjusted = x_values[:len(y_values)]

    CA3_density_per_branch_basal_plot.ax.plot(
        x_values_adjusted,
        y_values,
        color='lightgray',
        zorder=0
    )

# p_value_CA3 = pairwise_comparison(
#     CA3_density_basal_df[1],
#     CA3_density_basal_df[2])

plt.subplots_adjust(top=1., bottom=0.2)

# CA3_density_per_branch_basal_plot.save_figure(
#     saving_path / 'CA3_density_per_branch_basal.png',
#     dpi=dpi)

# %% 38) Z-scores traces of all neurons

fig, axes = plt.subplots(2, len(segmenter_list),
                         sharey=True,
                         figsize=(10, 3))

for n, segmenter in enumerate(segmenter_list):

    ts = segmenter_list[n].batch_ts_BLA[0][0]
    is_event_BLA = segmenter_list[n].calcium_events_BLA
    is_event_CA3 = segmenter_list[n].calcium_events_CA3

    probabilities_BLA = np.array(
        [sweep for spine in is_event_BLA for sweep in spine]
    )
    probabilities_BLA = probabilities_BLA[~np.isnan(probabilities_BLA)]
    dynamic_threshold_BLA = np.percentile(probabilities_BLA, 99)

    event_zscores_BLA = np.array([
        segmenter_list[n].batch_z_scores_BLA[i][j]
        for i in range(len(is_event_BLA))
        for j in range(len(is_event_BLA[i]))
        if is_event_BLA[i][j] >= dynamic_threshold_BLA])

    probabilities_CA3 = np.array(
        [sweep for spine in is_event_CA3 for sweep in spine]
    )
    probabilities_CA3 = probabilities_CA3[~np.isnan(probabilities_CA3)]
    dynamic_threshold_CA3 = np.percentile(probabilities_CA3, 99)

    event_zscores_CA3 = np.array([
        segmenter_list[n].batch_z_scores_CA3[i][j]
        for i in range(len(is_event_CA3))
        for j in range(len(is_event_CA3[i]))
        if is_event_CA3[i][j] >= dynamic_threshold_CA3])

    mean_BLA, ci_BLA = mean_and_ci(event_zscores_BLA)
    mean_CA3, ci_CA3 = mean_and_ci(event_zscores_CA3)

    axes[0][n].plot(
        ts,
        mean_BLA,
        color='black')

    axes[0][n].fill_between(
        ts,
        mean_BLA - ci_BLA,
        mean_BLA + ci_BLA,
        color=BLA_color,
        alpha=0.5)

    axes[1][n].plot(
        ts,
        mean_CA3,
        color='black')

    axes[1][n].fill_between(
        ts,
        mean_CA3 - ci_CA3,
        mean_CA3 + ci_CA3,
        color=CA3_color,
        alpha=0.5)

for n_row, row in enumerate(axes):
    for n_column, column in enumerate(row):
        column.spines[['bottom', 'top', 'right']].set_visible(False)
        column.set_xticks([])
        column.set_ylim(-1, 4)
        column.set_xlim(0, 4)

axes[-1][-1].plot([2, 3], [-1, -1], lw=1.5)
axes[-1][-1].text(
    x=2.5, y=-1.2, s="1 s",
    horizontalalignment='center', verticalalignment='top',
    fontsize=fontsize)

axes[1][0].set_ylabel('Z-score', fontsize=fontsize)


plt.subplots_adjust(wspace=0.05, hspace=0.25)
# plt.tight_layout()

# fig.savefig(
#     saving_path / 'Zscore_mean_and_CI_traces_BLA_and_CA3.png',
#     dpi=dpi)

# %% 39) Exploded-view of all branches and spines

landmarks = [100, 200, 300]

for neurites in spynalyzer_list:

    path = neurites.dataset.folder
    name = f"{path.parts[-3]}_{path.parts[-2]}"

    n_neurites = sum([1 if neurite else 0 for neurite in neurites])

    fig, axes = plt.subplots(
        n_neurites, 2,
        figsize=(8, 12),
        gridspec_kw={
            'hspace': None,
            'width_ratios': (3, 1)},
        layout="constrained")
    fig.suptitle(name, fontsize=fontsize)

    x_ticks = np.linspace(0, 300, 4, dtype=int)

    for n in range(n_neurites):

        if neurites[n]:
            color = (
                'red' if neurites[n].compartment == 'apical dendrite'
                else 'blue')

            neurites[n].plot(
                input_type='both',
                normalize=True, ax=axes[n][0], show_cbar=False,
                neurite_color=color,
                spine_size=20,
                spine_color='lightgray')

            interspine_distances_all = np.array(
                neurites[n].sequential_spine_distances())
            interspine_distances_BLA = np.array(
                neurites[n].sequential_spine_distances(input_type='BLA'))
            interspine_distances_CA3 = np.array(
                neurites[n].sequential_spine_distances(input_type='CA3'))

            if neurites[n].sequential_spine_distances():
                lambda_est = np.mean(interspine_distances_all)
                x = np.arange(
                    0, np.max(interspine_distances_all) + 1)
                poisson_pmf = poisson.pmf(x, mu=lambda_est)

                axes[n][1].plot(
                    x,
                    poisson_pmf,
                    # 'o-',
                    color='red')

                sns.histplot(
                    interspine_distances_all,
                    bins=20,
                    color='gray',
                    kde=False,
                    stat='probability',
                    # fill=False,
                    element='step',
                    ax=axes[n][1])
                sns.histplot(
                    interspine_distances_BLA,
                    bins=20,
                    color=BLA_color,
                    kde=False,
                    stat='probability',
                    # fill=False,
                    element='step',
                    ax=axes[n][1])
                sns.histplot(
                    interspine_distances_CA3,
                    bins=20,
                    color=CA3_color,
                    kde=False,
                    stat='probability',
                    # fill=False,
                    element='step',
                    ax=axes[n][1])

    for n, ax in enumerate(axes.flat):
        ax.set_aspect('auto')
        ax.spines[['top', 'right']].set_visible(False)
        ax.spines['bottom'].set_visible(False)
        ax.spines['left'].set_position(('outward', 20))
        ax.tick_params(axis='x', length=0)
        ax.set_xticklabels([])

    for n, ax in enumerate(axes[-1, :]):
        ax.spines['bottom'].set_visible(True)
        ax.spines['bottom'].set_position(('outward', 20))
        ax.tick_params(axis='x', labelsize=fontsize)

    for n, ax in enumerate(axes[:, 0]):
        ax.set_ylim(-20, 20)
        ax.set_yticks([-20., 0, 20.])
        ax.set_yticklabels([])
        ax.set_xticks(x_ticks)
        ax.set_xlim(0, x_ticks[-1])

        for landmark in landmarks:
            ax.axvline(
                x=landmark,
                color='gray',
                alpha=0.3,
                linestyle='dashed',
                linewidth=0.8,
                zorder=0,
                clip_on=False)

    for n, ax in enumerate(axes[:, 1]):
        ax.set_ylabel('')
        ax.set_yticklabels([])
        ax.set_ylim(0, 1.)
        ax.set_xlim(0, 150)
        ax.set_xticks(np.arange(0, 200, 50))

        if n != len(axes[:, 1]) - 1:
            ax.set_yticklabels([])

        else:
            ax.set_yticks([0, 1])
            ax.set_yticklabels([0, 1])
            ax.set_ylabel("Norm.\nfreq.", fontsize=fontsize)
            ax.tick_params(axis='y', labelsize=fontsize)

    axes[-1][0].set_xlabel("Length from origin (µm)", fontsize=fontsize)
    axes[-1][1].set_xlabel("Interspine dist. (µm)", fontsize=fontsize)
    axes[-1][0].tick_params(axis='x', length=3.5)
    axes[-1][1].tick_params(axis='x', length=3.5)

    axes[-1][0].set_xticklabels(x_ticks)
    axes[-1][1].set_xticklabels(np.arange(0, 200, 50))

    # fig.savefig(
    #     saving_path / f'{name}_aligned_neurites_BLA.png',
    #     dpi=dpi)

# %% bootstrap analysis

segmenter = segmenter_list[5]
dataset = dataset_list[5]

BLA_ratios_dict = {}
CA3_ratios_dict = {}
log_ratios_dict = {}

n_iterations = 1000
save_interval = 100

bla_filename = 'BLA_ratios_dict.h5'
ca3_filename = 'CA3_ratios_dict.h5'
log_ratios_filename = 'log_ratios_dict.h5'

fl.save(bla_filename, {})
fl.save(ca3_filename, {})
fl.save(log_ratios_filename, {})

for iteration in tqdm(range(n_iterations), desc="Iterations Progress"):

    random_prob_BLA, _ = segmenter.randomize_spines()
    random_prob_CA3, _ = segmenter.randomize_spines(input_type='CA3')

    neurites_copy_BLA = spa.SpyneAnalyzer(
        paths=dataset.paths,
        dataset=dataset,
        segmenter=segmenter,
        calcium_events_BLA=random_prob_BLA)
    neurites_copy_CA3 = spa.SpyneAnalyzer(
        paths=dataset.paths,
        dataset=dataset,
        segmenter=segmenter,
        calcium_events_BLA=random_prob_CA3)

    for neurite_index in range(neurites_copy_BLA.n_neurites):
        if neurites_copy_BLA[neurite_index]:
            bla_ratio = (
                neurites_copy_BLA[neurite_index].n_spines_BLA /
                neurites_copy_BLA.segmenter.n_spines_BLA) * 100

            if neurite_index not in BLA_ratios_dict:
                BLA_ratios_dict[neurite_index] = []

            BLA_ratios_dict[neurite_index].append(bla_ratio)

        if neurites_copy_CA3[neurite_index]:
            ca3_ratio = (
                neurites_copy_CA3[neurite_index].n_spines_CA3 /
                neurites_copy_CA3.segmenter.n_spines_CA3) * 100

            if neurite_index not in CA3_ratios_dict:
                CA3_ratios_dict[neurite_index] = []

            CA3_ratios_dict[neurite_index].append(ca3_ratio)

        if 'bla_ratio' in locals() and 'ca3_ratio' in locals():
            epsilon = 1e-10
            log_ratio = np.log((bla_ratio + epsilon) / (ca3_ratio + epsilon))
            if neurite_index not in log_ratios_dict:
                log_ratios_dict[neurite_index] = []
            log_ratios_dict[neurite_index].append(log_ratio)

    if (iteration + 1) % save_interval == 0:

        existing_bla_data = fl.load(bla_filename)
        existing_ca3_data = fl.load(ca3_filename)
        existing_log_ratios_data = fl.load(log_ratios_filename)

        for key, value in BLA_ratios_dict.items():
            if key in existing_bla_data:
                existing_bla_data[key].extend(value)
            else:
                existing_bla_data[key] = value
        fl.save(bla_filename, existing_bla_data)
        BLA_ratios_dict.clear()

        for key, value in CA3_ratios_dict.items():
            if key in existing_ca3_data:
                existing_ca3_data[key].extend(value)
            else:
                existing_ca3_data[key] = value
        fl.save(ca3_filename, existing_ca3_data)
        CA3_ratios_dict.clear()

        for key, value in log_ratios_dict.items():
            if key in existing_log_ratios_data:
                existing_log_ratios_data[key].extend(value)
            else:
                existing_log_ratios_data[key] = value
        fl.save(log_ratios_filename, existing_log_ratios_data)
        log_ratios_dict.clear()

existing_bla_data = fl.load(bla_filename)
for key, value in BLA_ratios_dict.items():
    if key in existing_bla_data:
        existing_bla_data[key].extend(value)
    else:
        existing_bla_data[key] = value
fl.save(bla_filename, existing_bla_data)

existing_ca3_data = fl.load(ca3_filename)
for key, value in CA3_ratios_dict.items():
    if key in existing_ca3_data:
        existing_ca3_data[key].extend(value)
    else:
        existing_ca3_data[key] = value
fl.save(ca3_filename, existing_ca3_data)

existing_log_ratios_data = fl.load(log_ratios_filename)
for key, value in log_ratios_dict.items():
    if key in existing_log_ratios_data:
        existing_log_ratios_data[key].extend(value)
    else:
        existing_log_ratios_data[key] = value
fl.save(log_ratios_filename, existing_log_ratios_data)

# %%
neuron_n = 2
segmenter = segmenter_list[neuron_n]
dataset = dataset_list[neuron_n]
analyzer = spynalyzer_list[neuron_n]
morph = morphology_list[neuron_n]

spine_global_index = {
    tuple(spine['centroid_fov']): idx  # Use a unique property like 'centroid_fov'
    for idx, spine in enumerate(segmenter.batch_spines_data)
}

spine_in_neurite = {}

for neurite_n in range(analyzer.n_neurites):
    if analyzer[neurite_n]:
        spine_in_neurite[neurite_n] = [
            spine_global_index.get(tuple(spine['centroid_fov']), None)
            for spine in analyzer[neurite_n].spines
        ]

spine_to_neurite = {}
for neurite_n, spine_indices in spine_in_neurite.items():
    for global_index in spine_indices:
        if global_index is not None:
            spine_to_neurite[global_index] = neurite_n

spine_dataframe = pd.DataFrame({
    'spine': [n for n, spine in enumerate(segmenter.batch_spines_data)],
    'centroid': [
        spine['centroid_fov'] for spine in segmenter.batch_spines_data],
    'branch': [
        spine_to_neurite.get(index, None)
        for index in range(len(segmenter.batch_spines_data))
    ],
    'BLA_event': [
        1 if sum(spine) > 0 else 0
        for spine in segmenter.calcium_events_binary_BLA],
    'CA3_event': [
        1 if sum(spine) > 0 else 0
        for spine in segmenter.calcium_events_binary_CA3]})
spine_dataframe.set_index('spine', inplace=True)

branch_event_sums = spine_dataframe.groupby('branch')[['BLA_event', 'CA3_event']].sum()
safe_branch_event_sums = branch_event_sums.copy()
safe_branch_event_sums['BLA_event'] = safe_branch_event_sums['BLA_event'].replace(0, 1e-9)
safe_branch_event_sums['CA3_event'] = safe_branch_event_sums['CA3_event'].replace(0, 1e-9)

safe_branch_event_sums['log_ratio'] = np.log10(
    safe_branch_event_sums['BLA_event'] / safe_branch_event_sums['CA3_event']
)
original_ratio = safe_branch_event_sums['log_ratio'].to_numpy()

n_neurites = spine_dataframe['branch'].nunique()
n_iterations = 10000

# iNdependent permutations
results_BLA = []
results_CA3 = []

for iteration in range(n_iterations):
    shuffled_bla_events = np.random.permutation(
        spine_dataframe['BLA_event'].values)

    temp_df = spine_dataframe.copy()
    temp_df['BLA_event_shuffled'] = shuffled_bla_events

    bla_event_counts_shuffled = temp_df.groupby('branch')[
        'BLA_event_shuffled'].sum()

    total_bla_events_shuffled = shuffled_bla_events.sum()

    bla_event_ratios_shuffled = (
        bla_event_counts_shuffled / total_bla_events_shuffled)

    for branch, ratio in bla_event_ratios_shuffled.items():
        results_BLA.append(ratio)

results_BLA_array = np.array(results_BLA).reshape(n_iterations, n_neurites).T
safe_BLA_array = np.where(results_BLA_array == 0, 1e-9, results_BLA_array)

for iteration in range(n_iterations):
    shuffled_ca3_events = np.random.permutation(
        spine_dataframe['CA3_event'].values)

    temp_df = spine_dataframe.copy()
    temp_df['CA3_event_shuffled'] = shuffled_ca3_events

    ca3_event_counts_shuffled = temp_df.groupby('branch')[
        'CA3_event_shuffled'].sum()

    total_ca3_events_shuffled = shuffled_ca3_events.sum()

    ca3_event_ratios_shuffled = (
        ca3_event_counts_shuffled / total_ca3_events_shuffled)

    for branch, ratio in ca3_event_ratios_shuffled.items():
        results_CA3.append(ratio)

results_CA3_array = np.array(results_CA3).reshape(n_iterations, n_neurites).T
safe_CA3_array = np.where(results_CA3_array == 0, 1e-9, results_CA3_array)

ratio_array = (safe_BLA_array / safe_CA3_array)
results_log_BLA_CA3 = np.log10(ratio_array)
log_values = results_log_BLA_CA3.flatten()

# Synchronized permutation
results_BLA_CA3 = []

for iteration in range(n_iterations):
    # Generate a random permutation for BLA_event and CA3_event

    perm_indices = np.random.permutation(len(spine_dataframe))

    # Shuffle BLA_event and CA3_event columns based on the same permutation
    shuffled_bla_events = spine_dataframe['BLA_event'].values[perm_indices]
    shuffled_ca3_events = spine_dataframe['CA3_event'].values[perm_indices]

    # Create a temporary DataFrame with shuffled columns
    temp_df = spine_dataframe.copy()
    temp_df['BLA_event_shuffled'] = shuffled_bla_events
    temp_df['CA3_event_shuffled'] = shuffled_ca3_events

    # Group by branch and calculate the sum of shuffled BLA_event and CA3_event
    shuffled_sums = temp_df.groupby('branch')[
        ['BLA_event_shuffled', 'CA3_event_shuffled']].sum()

    # Replace zeros with a small constant to avoid division by zero
    shuffled_sums['BLA_event_shuffled'] = shuffled_sums[
        'BLA_event_shuffled'].replace(0, 1e-9)
    shuffled_sums['CA3_event_shuffled'] = shuffled_sums[
        'CA3_event_shuffled'].replace(0, 1e-9)

    # Calculate the total for normalization
    total_BLA = shuffled_sums['BLA_event_shuffled'].sum()
    total_CA3 = shuffled_sums['CA3_event_shuffled'].sum()

    # Calculate log ratios for each branch
    log_ratios = np.log10(
        (shuffled_sums['BLA_event_shuffled'] / total_BLA) /
        (shuffled_sums['CA3_event_shuffled'] / total_CA3))

    # Store the results for this iteration
    results_BLA_CA3.append(log_ratios.values)

# Convert results to a numpy array
results_BLA_CA3_array = np.array(results_BLA_CA3).T
log_values_synchronized = results_BLA_CA3_array.flatten()

fig, ax = plt.subplots(figsize=(10, 8), layout="constrained")

for branch_idx in range(results_BLA_CA3_array.shape[0]):
    branch_values = results_BLA_CA3_array[branch_idx, :]
    # sns.kdeplot(
    #     branch_values, label=f'Branch {branch_idx}', linewidth=2, ax=ax)
    ax.axvline(
        original_ratio[branch_idx], linestyle='--', linewidth=1.5, color='gray')

sns.kdeplot(log_values_synchronized, fill=False, color='black', alpha=0.6, linewidth=2, label='All Branches', ax=ax)

ax.spines[['top', 'right']].set_visible(False)
ax.spines[['bottom', 'left']].set_position(('outward', 20))

ax.set_xlabel('log(BLA/CA3)', fontsize=12)
ax.set_ylabel('Density', fontsize=12)

ax.legend(fontsize=10, title='Branches')

# %%

colors = sns.color_palette("husl", len(spynalyzer_list))

for i, (analyzer, segmenter) in enumerate(zip(
        spynalyzer_list, segmenter_list)):

    path = analyzer.dataset.folder
    name = f"{path.parts[-3]}_{path.parts[-2]}"

    bla_total = sum(
        [1 if sum(spine) > 0 else 0
         for spine in segmenter.calcium_events_binary_BLA])
    ca3_total = sum(
        [1 if sum(spine) > 0 else 0
         for spine in segmenter.calcium_events_binary_CA3])

    bla_ratios = []
    ca3_ratios = []

    fig, ax = plt.subplots(figsize=(5, 4), layout="constrained")

    non_none_neurites = [
        neurite for neurite in analyzer if neurite is not None]

    for neurite_n, neurite in enumerate(non_none_neurites):

        if neurite.spines:
            bla_ratio = neurite.n_spines_BLA / bla_total
            ca3_ratio = neurite.n_spines_CA3 / ca3_total
        else:
            bla_ratio = 0
            ca3_ratio = 0

        bla_ratios.append(bla_ratio)
        ca3_ratios.append(ca3_ratio)

        ax.scatter(
            bla_ratio,
            ca3_ratio,
            facecolor='none',
            edgecolor=colors[i],
            clip_on=False,
            lw=2,
            s=60,
            alpha=0.5,)

    bla_ratios = np.array(bla_ratios)
    ca3_ratios = np.array(ca3_ratios)

    overall_max = np.max(bla_ratios)

    pearson_corr, _ = pearsonr(bla_ratios, ca3_ratios)
    r_squared = pearson_corr**2

    slope, intercept = np.polyfit(bla_ratios, ca3_ratios, 1)
    x_vals = np.linspace(
        0, overall_max, 100)
    y_vals = slope * x_vals + intercept

    n = len(bla_ratios)
    y_fit = slope * bla_ratios + intercept
    residuals = ca3_ratios - y_fit
    residual_sum_of_squares = np.sum(residuals**2)
    mean_x = np.mean(bla_ratios)
    t_value = t.ppf(0.975, df=n-2)  # Two-tailed t-value for 95% CI

    se = np.sqrt(residual_sum_of_squares / (n - 2))
    se_y = se * np.sqrt(
        1/n + (x_vals - mean_x)**2 / np.sum((bla_ratios - mean_x)**2))

    y_upper = y_vals + t_value * se_y
    y_lower = y_vals - t_value * se_y

    ax.plot(
        x_vals,
        y_vals,
        color='gray',
        # clip_on=False,
        label=f'$R^2 = {r_squared:.2f}$',
        )

    ax.fill_between(
        x_vals,
        y_lower,
        y_upper,
        edgecolor='none',
        facecolor=colors[i],
        alpha=0.2,
        label='95% CI')

    ax.set_aspect('equal')
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['bottom', 'left']].set_position(("outward", 20))
    ax.set_xlabel('BLA branch / BLA total', fontsize=fontsize)
    ax.set_ylabel('CA3 branch / CA3 total', fontsize=fontsize)
    ax.set_xticks([0, 0.2, 0.4])
    ax.set_yticks([0, 0.2, 0.4])
    ax.tick_params('both', labelsize=fontsize)
    ax.set_xlim(0, 0.4)
    ax.set_ylim(0, 0.4)
    ax.legend(fontsize=fontsize, loc='upper left', frameon=False)

    # fig.savefig(
    #     saving_path / f'{name}_spine_ratio_linear_regression.png',
    #     dpi=dpi)

# %%

neuron_n = 6
segmenter = segmenter_list[neuron_n]
dataset = dataset_list[neuron_n]
analyzer = spynalyzer_list[neuron_n]
morph = morphology_list[neuron_n]

for neurite_n in range(analyzer.n_neurites):

    path = analyzer.dataset.folder
    name = f"{path.parts[-3]}_{path.parts[-2]}"

    if analyzer[neurite_n] and analyzer[neurite_n].spines:

        fig, axes = plt.subplots(
            2, 1,
            figsize=(8, 5),
            layout="constrained",
            gridspec_kw={'height_ratios': (2, 1)})

        # Data for BLA
        k_bla = analyzer[neurite_n].n_spines_BLA
        n = analyzer[neurite_n].n_spines
        p_bla = segmenter.n_spines_BLA / len(segmenter.batch_spines_data)

        mean_bla, var_bla, skew_bla, kurt_bla = binom.stats(
            n, p_bla, moments='mvsk')

        x_bla = np.arange(0, 15)
        #     binom.ppf(0.01, n, p_bla),
        #     binom.ppf(0.99, n, p_bla)
        # )
        result_bla = binomtest(k_bla, n, p_bla, alternative='two-sided')
        print(f"BLA Neurite {neurite_n}, p-value: {result_bla.pvalue}")

        axes[0].plot(
            x_bla,
            binom.pmf(x_bla, n, p_bla),
            color=BLA_color,
            lw=5,
            alpha=0.5)
        axes[0].axvline(
            k_bla,
            color=BLA_color,
            linestyle='--',
            label=f'Observed: {k_bla}')
        if result_bla.pvalue < 0.06:
            pvalue_text = (
                "< 0.001" if result_bla.pvalue < 0.001
                else f"{round(result_bla.pvalue, 3)}")
            axes[0].text(
                k_bla + 0.3,
                np.max(binom.pmf(x_bla, n, p_bla)),
                s=f"p = {pvalue_text}",
                color=BLA_color,
                fontsize=fontsize
            )

        # Data for CA3
        k_ca3 = analyzer[neurite_n].n_spines_CA3
        p_ca3 = segmenter.n_spines_CA3 / len(segmenter.batch_spines_data)

        mean_ca3, var_ca3, skew_ca3, kurt_ca3 = binom.stats(
            n, p_ca3, moments='mvsk')

        x_ca3 = np.arange(0, 15)
        #     binom.ppf(0.01, n, p_ca3),
        #     binom.ppf(0.99, n, p_ca3)
        # )
        result_ca3 = binomtest(k_ca3, n, p_ca3, alternative='two-sided')
        print(f"CA3 Neurite {neurite_n}, p-value: {result_ca3.pvalue}")

        axes[0].plot(
            x_ca3,
            binom.pmf(x_ca3, n, p_ca3),
            color=CA3_color,
            lw=5,
            alpha=0.5)
        axes[0].axvline(
            k_ca3,
            color=CA3_color,
            linestyle='--',
            label=f'Observed: {k_ca3}')
        axes[0].legend()
        if result_ca3.pvalue < 0.06:
            pvalue_text = (
                "< 0.001" if result_ca3.pvalue < 0.001
                else f"{round(result_ca3.pvalue, 3)}")
            axes[0].text(
                k_ca3 + 0.3,
                np.max(binom.pmf(x_ca3, n, p_ca3)),
                s=f"p = {pvalue_text}",
                color=CA3_color,
                fontsize=fontsize,
            )
        axes[0].legend(fontsize=fontsize)

        axes[0].set_ylim(0, 0.4)
        axes[0].set_xlim(0, 14)
        axes[0].set_xticks(np.arange(0, 16, 2))
        axes[0].spines['left'].set_position(('outward', 20))
        axes[0].set_xlabel(
            r'N of spines with Ca$^{+2}$ event', fontsize=fontsize)
        axes[0].set_ylabel('Probability', fontsize=fontsize)

        # Neurite plot
        analyzer[neurite_n].plot(
            normalize=True, ax=axes[1], show_cbar=False)

        axes[1].set_aspect('equal')
        axes[1].spines[['bottom', 'left']].set_position(('outward', 20))
        axes[1].set_ylim(-20, 20)
        axes[1].set_xlim(0, 300)
        axes[1].set_xlabel('Distance from origin (µm)', fontsize=fontsize)
        axes[1].set_yticklabels([])

        # if neurite_n == 8 or neurite_n == 12 or neurite_n == 14:
        #     fig.savefig(
        #         saving_path / f"{name}_neurite{neurite_n}_binomial_dist.png",
        #         dpi=dpi)

# %%

for analyzer in spynalyzer_list:

    fig, axes = plt.subplots(
        1, 2,
        figsize=(6, 4),
        layout="constrained",
        sharex=True,
        sharey=True)

    all_x_bla, all_y_bla = [], []
    all_x_ca3, all_y_ca3 = [], []

    for neurite_n in range(analyzer.n_neurites):

        path = analyzer.dataset.folder
        name = f"{path.parts[-3]}_{path.parts[-2]}"

        if analyzer[neurite_n] and analyzer[neurite_n].spines:

            # Data for BLA
            k_bla = analyzer[neurite_n].n_spines_BLA
            n = analyzer[neurite_n].n_spines
            p_bla = segmenter.n_spines_BLA / len(segmenter.batch_spines_data)

            x_bla = np.argmax(binom.pmf(np.arange(0, 15), n, p_bla))
            y_bla = k_bla
            all_x_bla.append(x_bla)
            all_y_bla.append(y_bla)

            axes[0].scatter(
                x_bla,
                y_bla,
                edgecolor=BLA_color,
                facecolor='none',
                s=60,
                lw=1.5,
                alpha=0.5,
                clip_on=False)

            # Data for CA3
            k_ca3 = analyzer[neurite_n].n_spines_CA3
            p_ca3 = segmenter.n_spines_CA3 / len(segmenter.batch_spines_data)

            x_ca3 = np.argmax(binom.pmf(np.arange(0, 15), n, p_ca3))
            y_ca3 = k_ca3
            all_x_ca3.append(x_ca3)
            all_y_ca3.append(y_ca3)

            axes[1].scatter(
                x_ca3,
                y_ca3,
                edgecolor=CA3_color,
                facecolor='none',
                s=60,
                lw=1.5,
                alpha=0.5,
                clip_on=False)

    if all_x_bla and all_y_bla:
        r_bla, p_bla_corr = pearsonr(all_x_bla, all_y_bla)
        slope_bla, intercept_bla, _, _, std_err_bla = linregress(
            all_x_bla, all_y_bla)

        x_line = np.linspace(0, max(all_x_bla), 100)
        y_line = slope_bla * x_line + intercept_bla
        ci_bla = 1.96 * std_err_bla  # 95% CI

        axes[0].plot(
            x_line,
            y_line,
            color=BLA_color,
            lw=1.5,
            label=f'r = {r_bla:.2f}')
        axes[0].fill_between(
            x_line,
            y_line - ci_bla,
            y_line + ci_bla,
            color=BLA_color,
            alpha=0.2,
        )

    # Pearson correlation and regression for CA3
    if all_x_ca3 and all_y_ca3:
        r_ca3, p_ca3_corr = pearsonr(all_x_ca3, all_y_ca3)
        slope_ca3, intercept_ca3, _, _, std_err_ca3 = linregress(
            all_x_ca3, all_y_ca3)

        x_line = np.linspace(0, max(all_x_ca3), 100)
        y_line = slope_ca3 * x_line + intercept_ca3
        ci_ca3 = 1.96 * std_err_ca3  # 95% CI

        axes[1].plot(
            x_line,
            y_line,
            color=CA3_color,
            lw=1.5,
            label=f'r = {r_ca3:.2f}')
        axes[1].fill_between(
            x_line,
            y_line - ci_ca3,
            y_line + ci_ca3,
            color=CA3_color,
            alpha=0.2,
        )

    axes[0].set_xlabel(
        'Expected N of\n' + r'spines with Ca$^{+2}$ event',
        fontsize=fontsize)
    axes[0].set_ylabel(
        'Observed N of\n' + r'spines with Ca$^{+2}$ event',
        fontsize=fontsize)

    for ax in axes:
        ax.legend(loc='best', fontsize=fontsize)
        ax.set_ylim(0, 14)
        ax.set_xlim(0, 10)
        ax.set_xticks(np.arange(0, 12, 2))
        ax.set_yticks(np.arange(0, 16, 2))
        ax.spines[['left', 'bottom']].set_position(('outward', 20))
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params('both', labelsize=fontsize)

    # fig.savefig(
    #        saving_path / f"{name}_expected_vs_observed.png",
    #        dpi=dpi)



