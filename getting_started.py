""" Created on Sun Sep 24 14:51:36 2023
    @author: dcupolillo """

import spyne
from neuronpath.path import neuronpath

# Choose the structure to be analyzed
paths = neuronpath('240813', 1)

# Initialize the dataset and segmenter to identify spines
dataset = spyne.ImagingDataset(paths)
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

# Work on single roi
segmented_roi = segmenter[0]
segmented_roi.inference()

# Display inference result and analisys
segmented_roi.plot_masks()
segmented_roi.plot_dFF()
segmented_roi.plot_zscore()

# Index individual spines
spine = segmented_roi[0]

# To launch a batch analysis
segmenter.collect_all_data()

# To launch multiple batch analyses
neuronpaths_list = [
    neuronpath('240828', 1),
    neuronpath('240910', 1),
    neuronpath('240912', 2),
    neuronpath('240913', 1),
]
spyne.run_multiple_batches(neuronpaths_list)

# Not at 16 hz
not_16hx_list = [
    neuronpath('240813', 1),
    neuronpath('240814', 2),
    neuronpath('240827', 2),
]
spyne.run_multiple_batches(not_16hx_list)

neuronpaths_list = [
    neuronpath('240813', 1),
    neuronpath('240814', 2),
    neuronpath('240827', 2),
    neuronpath('240828', 1),
    neuronpath('240910', 1),
    neuronpath('240912', 2),
    neuronpath('240913', 1),
]
