""" Created on Mon Oct 28 17:23:20 2024
    @author: dcupolillo """

import flammkuchen as fl
import matplotlib.pyplot as plt
import numpy as np
import random
from itertools import chain

#%% Re-organiza a h5 file


def manage_h5(h5_file):

    labels = h5_file['binary_decision']
    excluded_traces = h5_file['excluded_traces']
    included_traces = h5_file['included_traces']
    timestamps = h5_file['timestamps']

    included_counter = 0
    excluded_counter = 0

    z_scores = []
    selected_timestamps = []

    for label in labels:
        if label == 0:
            z_scores.append(excluded_traces[excluded_counter])
            selected_timestamps.append(timestamps[excluded_counter])
            excluded_counter += 1
        else:
            z_scores.append(included_traces[included_counter])
            selected_timestamps.append(timestamps[included_counter])
            included_counter += 1

    h5_dictionary = {
        'label': np.array(labels),
        'ts': np.array(selected_timestamps, dtype=np.float32),
        'zscore': np.array(z_scores, dtype=np.float32)
    }

    return h5_dictionary

# Usage
file_to_load = r"trace_data2.h5"
output_filename = r"zscore_label4.h5"

h5_file = fl.load(file_to_load)
h5_dictionary = manage_h5(h5_file)

fl.save(output_filename, h5_dictionary)

# %% Take only traces labeled as '1'

def filter_h5(h5_file, label_to_filter):

    labels = h5_file['label']
    timestamps = h5_file['ts']
    z_scores = h5_file['zscore']

    filtered_indices = (labels == label_to_filter)

    filtered_labels = labels[filtered_indices]
    filtered_timestamps = timestamps[filtered_indices]
    filtered_z_scores = z_scores[filtered_indices]

    h5_dictionary = {
        'label': filtered_labels,
        'ts': filtered_timestamps,
        'zscore': filtered_z_scores
    }

    return h5_dictionary

# Usage
file_to_load = r"zscore_label4.h5"
output_filename = r"filtered_zscore_labels4.h5"

file = fl.load(file_to_load)
h5_dictionary = filter_h5(file, 1)
fl.save(output_filename, h5_dictionary)

# %% Merge h5 files

def merge_h5(h5_file_1, h5_file_2):

    merged_labels = np.concatenate(
        [h5_file_1['label'],
         h5_file_2['label']])
    merged_timestamps = np.concatenate(
        [h5_file_1['ts'],
         h5_file_2['ts']])
    merged_z_scores = np.concatenate(
        [h5new 1-labeled traces_file_1['zscore'],
         h5_file_2['zscore']])


    h5_dictionary = {
        'label': merged_labels,
        'ts': merged_timestamps,
        'zscore': merged_z_scores
    }

    return h5_dictionary

# Usage
file_1 = fl.load(r"neuralnetwork/zscore_decoder/datasets/1-labeled_zscore_only.h5")
file_2 = fl.load(r"neuralnetwork/zscore_decoder/datasets/zscore_labels.h5")
output_filename = r"neuralnetwork/zscore_decoder/datasets/bigger_zscore_labels.h5"

h5_dictionary = merge_h5(file_1, file_2)
fl.save(output_filename, h5_dictionary)

# %%

def check_identical_traces(h5_file):

    z_scores = h5_file['zscore']
    traces_as_tuples = [tuple(trace) for trace in z_scores]

    unique_traces = set()
    duplicates_index = []
    unique_index = []
    
    for i, trace in enumerate(traces_as_tuples):
        if trace in unique_traces:
            duplicates_index.append(i)
        else:
            unique_traces.add(trace)
            unique_index.append(i)

    return unique_index


def remove_duplicates(h5_file):
    # Get indices of unique traces
    unique_index = check_identical_traces(h5_file)

    # Filter the arrays using unique indices
    unique_labels = h5_file['label'][unique_index]
    unique_timestamps = h5_file['ts'][unique_index]
    unique_z_scores = h5_file['zscore'][unique_index]

    # Create new dictionary with unique entries
    unique_data = {
        'label': unique_labels,
        'ts': unique_timestamps,
        'zscore': unique_z_scores
    }

    return unique_data

# Usage
file = fl.load(r"zscore_labels.h5")
unique_file = remove_duplicates(file)

fl.save("unique_zscore_labels.h5", unique_file)

# %% Examine a dataset

def examine_h5(h5_file):

    total_traces = len(h5_file['label'])
    zero_traces = (h5_file['label'] == 0).sum()
    one_traces = (h5_file['label'] == 1).sum()

    print(f"Number of entries: {total_traces}")
    print(f"Number of 0-labeled entries: {zero_traces}")
    print(f"Number of 1-labeled entries: {one_traces}")

    fig, axes = plt.subplots(2, 1, figsize=(3, 5), sharex=True, sharey=True)

    axes[0].set_ylabel(f"0-labeled:\n{zero_traces} / {total_traces}")
    axes[1].set_ylabel(f"1-labeled:\n{one_traces} / {total_traces}")

    zero_label_traces = [h5_file['zscore'][i] for i in range(total_traces) if h5_file['label'][i] == 0]
    one_label_traces = [h5_file['zscore'][i] for i in range(total_traces) if h5_file['label'][i] == 1]

    mean_zero = np.mean(zero_label_traces, axis=0)
    mean_one = np.mean(one_label_traces, axis=0)

    for i, label in enumerate(h5_file['label']):
        if label == 0:
            axes[0].plot(h5_file['ts'][i], h5_file['zscore'][i], color='gray', alpha=0.3)
        else:
            axes[1].plot(h5_file['ts'][i], h5_file['zscore'][i], color='gray', alpha=0.3)
    
    # Overlay means
    axes[0].plot(h5_file['ts'][0], mean_zero, color='darkblue', linewidth=2, label='Mean 0-labeled')
    axes[1].plot(h5_file['ts'][0], mean_one, color='darkred', linewidth=2, label='Mean 1-labeled')

    plt.tight_layout()

# Usage
file = fl.load(r"balanced_zscore_labels.h5")
examine_h5(file)

# %% Balance 0 and 1 labels

def balance_labels(h5_file):

    labels = h5_file['label']
    timestamps = h5_file['ts']
    z_scores = h5_file['zscore']

    one_indices = np.where(labels == 1)[0]
    zero_indices = np.where(labels == 0)[0]

    num_ones = len(one_indices)

    if num_ones < len(zero_indices):
        sampled_zero_indices = np.random.choice(zero_indices, num_ones, replace=False)
    else:
        sampled_zero_indices = zero_indices

    balanced_indices = np.concatenate([one_indices, sampled_zero_indices])

    np.random.shuffle(balanced_indices)

    balanced_data = {
        'label': labels[balanced_indices],
        'ts': timestamps[balanced_indices],
        'zscore': z_scores[balanced_indices]
    }
    
    return balanced_data

# Usage
file = fl.load(r"zscore_labels.h5")
balanced_file = balance_labels(file)

fl.save("balanced_zscore_labels.h5", balanced_file)


# %% Take new 1-labeled traces

# Initialize lists to store combined traces and timestamps for all segmenters
all_traces_combined = []
all_ts_combined = []

for segmenter in segmenter_list:
    # Flatten the BLA and CA3 predictions
    predictions_BLA = list(chain.from_iterable(segmenter.calcium_events_binary_BLA))
    predictions_CA3 = list(chain.from_iterable(segmenter.calcium_events_binary_CA3))

    # Get indices where predictions > 0
    indices_BLA = [n for n, pred in enumerate(predictions_BLA) if pred > 0]
    indices_CA3 = [n for n, pred in enumerate(predictions_CA3) if pred > 0]

    # Ensure non-overlapping random indices
    try:
        random_indices_BLA = random.sample(indices_BLA, 60)
        random_indices_CA3 = random.sample(
            [idx for idx in indices_CA3 if idx not in random_indices_BLA], 60
        )
    except ValueError:
        random_indices_BLA = random.sample(indices_BLA, 40)
        random_indices_CA3 = random.sample(
            [idx for idx in indices_CA3 if idx not in random_indices_BLA], 40
        )

    # Flatten calcium event traces and timestamps
    calcium_events_BLA = np.array(list(chain.from_iterable(segmenter.batch_z_scores_BLA)))
    batch_ts_BLA = np.array(list(chain.from_iterable(segmenter.batch_ts_BLA)))
    calcium_events_CA3 = np.array(list(chain.from_iterable(segmenter.batch_z_scores_CA3)))
    batch_ts_CA3 = np.array(list(chain.from_iterable(segmenter.batch_ts_CA3)))

    # Extract traces and timestamps using indices
    traces_BLA = calcium_events_BLA[random_indices_BLA]
    ts_BLA = batch_ts_BLA[random_indices_BLA]

    traces_CA3 = calcium_events_CA3[random_indices_CA3]
    ts_CA3 = batch_ts_CA3[random_indices_CA3]

    # Combine the traces and timestamps for this segmenter
    traces_combined = np.concatenate((traces_BLA, traces_CA3), axis=0)
    ts_combined = np.concatenate((ts_BLA, ts_CA3), axis=0)

    # Append to global lists
    all_traces_combined.append(traces_combined)
    all_ts_combined.append(ts_combined)

# Concatenate all segmenter results into final arrays
final_traces_combined = np.concatenate(all_traces_combined, axis=0)
final_ts_combined = np.concatenate(all_ts_combined, axis=0)

# Create combined labels
num_samples = final_traces_combined.shape[0]  # Total number of samples
labels = np.ones(num_samples, dtype=np.int32)  # All labels set to 1

# Convert traces and timestamps to appropriate NumPy arrays
zscore = np.array(final_traces_combined, dtype=np.float32)  # Z-score traces
ts = np.array(final_ts_combined, dtype=np.float32)          # Timestamps

# Create the structured dictionary
data_dict = {
    'label': labels,
    'ts': ts,
    'zscore': zscore
}

fl.save("1-labeled_zscore_only.h5", data_dict)
