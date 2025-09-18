""" Created on Sun Sep 24 14:51:36 2023
    @author: dcupolillo """

import spyne
from pathlib import Path

date = "250820"
cell_n = "cell0001"
data_folder = Path("data")
imaging_folder = data_folder / date / cell_n / "neuron"

# ======================
# Dataset initialization
# ======================

dataset = spyne.ImagingDataset(imaging_folder)
ephy = spyne.EphyDataset(dataset)
segmenter = spyne.DatasetSegmenter(dataset)

# Index individual sweep of selected ROI of slected Scanfield
# Dimensions are: [roi_number][sweep_number]
roi = dataset[0]
sweep = roi[0]

# Separate channels of individual ROI
channel_green = sweep.ch1
channel_red = sweep.ch2

# display both channels or single channel
sweep.show()
channel_red.show()

# Display individual frames
single_frame_green = channel_green[0]
single_frame_green.show(cmap='binary')

# To launch an analysis on a complete neuron dataset
segmenter.collect_all_data()

# Work on single roi after data collection
segmented_roi = segmenter[0]

# ================
# Plotting example
# ================

import spyne.plot as plt_spyne

# Plot all spine 2D position
plt_spyne.scatter(segmenter.spines_data)

# Plot BLA-activated spines
plt_spyne.scatter_event(
    segmenter.spines_data,
    segmenter.calcium_events_binary_BLA)

# Traces heatmap
plt_spyne.heatmap(
    segmenter.dFF_BLA,
    segmenter.ts_BLA,
    segmenter.calcium_events_binary_BLA)