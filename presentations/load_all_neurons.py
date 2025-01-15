""" Created on Wed Oct 30 10:17:24 2024
    @author: dcupolillo """

from neuronpath.path import neuronpath
import spynalyzer as spa
import actionpytential as ap
from pathlib import Path
import numpy as np
import flammkuchen as fl


def initialize_neuronpaths():
    return [
        neuronpath('240813', 1),
        neuronpath('240814', 2),
        neuronpath('240827', 2),
        neuronpath('240828', 1),
        neuronpath('240910', 1),
        neuronpath('240912', 2),
        neuronpath('240913', 1),
    ]


def prepare_spynalyzer_and_traces():

    neuronpaths_list = initialize_neuronpaths()

    spynalyzer_list = [spa.SpyneAnalyzer(path) for path in neuronpaths_list]

    traces_list = [ap.EphyDataset(path) for path in neuronpaths_list]

    morphology_list, scanfields_list, dataset_list, segmenter_list = zip(*[
        (element.morph, element.sf, element.dataset, element.segmenter)
        for element in spynalyzer_list
    ])

    return (
        spynalyzer_list,
        traces_list,
        morphology_list,
        scanfields_list,
        dataset_list,
        segmenter_list)


def prepare_classification(
        dataset_list,
        segmenter_list,
):
    (
         paths,
         batch_zscores,
         batch_ts,
         zscore_data,
         event_probabilities
     ) = {}, {}, {}, {}, {}

    (
         binary_event,
         dynamic_thresholds,
         difference_array
     ) = {}, {}, {}

    for idx, (dataset, segmenter) in enumerate(zip(
            dataset_list, segmenter_list)):

        paths[idx] = {
            region: Path(
                dataset.folder.parent /
                "zscore_threshold_method" / f"calcium_event_{region}.h5"
            ) for region in ["BLA", "CA3"]
        }

        batch_zscores[idx] = {
            "BLA": segmenter.batch_z_scores_BLA,
            "CA3": segmenter.batch_z_scores_CA3
        }

        batch_ts[idx] = {
            "BLA": segmenter.batch_ts_BLA,
            "CA3": segmenter.batch_ts_CA3
        }

        zscore_data[idx] = {
            region: fl.load(paths[idx][region])
            for region in ["BLA", "CA3"]
        }

        event_probabilities[idx] = {
            "BLA": segmenter.calcium_events_BLA,
            "CA3": segmenter.calcium_events_CA3
        }

        dynamic_thresholds[idx] = {
            region: np.percentile(
                np.array([
                    sweep for spine in event_probabilities[idx][region]
                    for sweep in spine if not np.isnan(sweep)]), 99)
            for region in ["BLA", "CA3"]
        }

        binary_event[idx] = {
            region: np.zeros_like(event_probabilities[idx][region])
            for region in ["BLA", "CA3"]
        }

        for region in ["BLA", "CA3"]:
            for i, spine in enumerate(event_probabilities[idx][region]):
                for n, sweep in enumerate(spine):
                    if (sweep >= dynamic_thresholds[idx][region]):
                        binary_event[idx][region][i][n] = 1

        difference_array[idx] = {
                region: np.abs(zscore_data[idx][region] -
                               binary_event[idx][region])
                for region in ["BLA", "CA3"]
        }

    return (
        paths,
        batch_zscores,
        batch_ts,
        zscore_data,
        event_probabilities,
        binary_event,
        dynamic_thresholds,
        difference_array)
