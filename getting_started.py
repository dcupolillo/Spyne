""" Created on Sun Sep 24 14:51:36 2023
    @author: dcupolillo """

import spyne
from pathlib import Path

date = "250820"
cell_n = "cell0001"
data_folder = Path("data")
imaging_folder = data_folder / date / cell_n / "neuron"

# Initialize the dataset and segmenter to identify spines
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
