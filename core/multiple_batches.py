""" Created on Mon Sep 16 10:43:21 2024
    @author: dcupolillo """

import spyne
from neuronpath.path import neuronpath
from tqdm import tqdm


def run_multiple_batches(
        neuronpaths: list
) -> None:

    if not isinstance(neuronpaths, list):
        raise ValueError(f"{neuronpaths} is not a list.")

    # if not all(isinstance(n, neuronpath) for n in neuronpaths):
    #     raise ValueError("neuronpath list elements should be neuronpaths.")

    for n, path in enumerate(neuronpaths):

        dataset = spyne.ImagingDataset(path)
        segmenter = spyne.DatasetSegmenter(dataset)

        segmenter.collect_all_data()
