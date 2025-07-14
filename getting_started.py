""" Created on Sun Sep 24 14:51:36 2023
    @author: dcupolillo """

import spyne
from pathlib import Path

date = "250604"
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

# Work on single roi
segmented_roi = segmenter[0]
segmented_roi.inference()

# Display inference result and analisys
segmented_roi.plot_masks()
segmented_roi.plot_dFF()
segmented_roi.plot_zscore()

# Index individual spines
spine = segmented_roi[0]

# To launch an analysis on a complete neuron dataset
segmenter.collect_all_data()

test_model_path = Path(
    r"C:\Users\dcupolillo\Projects\spyne",
    r"neuralnetwork\spine_segmentation\bayesian_search",
    r"models\model_250414_trial18.h5")

segmenter = spyne.DatasetSegmenter(
    dataset,
    segmentation_model_fn=test_model_path)




dates = [_dir.stem for _dir in data_folder.iterdir()]
cell_ns = [d.stem for date in dates for d in (data_folder / date).iterdir()]

dataset_list = [None] * len(dates)
segmenter_list = [None] * len(dates)

for n, (date, cell_n) in enumerate(zip(dates, cell_ns)):
    # if dataset_list[n] is not None:
    #     continue
    # print(date)
    imaging_folder = data_folder / date / cell_n / "neuron"
    saving_folder = imaging_folder.parent / "time_series" / "3x3x3_median_filter"
    dataset = spyne.ImagingDataset(imaging_folder, kernel_size=(3, 3, 3))
    dataset_list[n] = dataset
    segmenter = spyne.DatasetSegmenter(dataset)
    segmenter.collect_all_data(save_path=saving_folder)
    segmenter_list[n] = segmenter
