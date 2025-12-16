""" Created on Sun Sep 24 14:51:36 2023
    @author: dcupolillo """

import spyne
from pathlib import Path

date = "240827"
cell_n = "cell0002"
data_folder = Path("data")
imaging_folder = data_folder / date / cell_n

# ======================
# Dataset initialization
# ======================

dataset = spyne.ImagingDataset(imaging_folder)
ephy = spyne.EphyDataset(dataset)
spine_dataset = spyne.SpineDataset(dataset)

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
spine_dataset.collect_all_data()

# Work on single roi after data collection
segmented_roi = spine_dataset[0]

# ================
# Plotting example
# ================

import spyne.plot

# Plot all spine 2D position
spyne.plot.scatter(spine_dataset.spines_data)

# Plot BLA-activated spines
spyne.plot.scatter_event(
    spine_dataset.spines_data,
    spine_dataset.calcium_events_binary_BLA)

# Traces heatmap
spyne.plot.heatmap(
    spine_dataset.dFF_BLA,
    spine_dataset.ts_BLA,
    spine_dataset.calcium_events_binary_BLA)
