""" Created on Tue Feb 13 17:11:49 2024
    @author: dcupolillo """

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.keras.optimizers import Adam
import segmentation_models as sm
import time
from datetime import datetime
import flammkuchen as fl
import pandas as pd
import random

from deepd3.model import DeepD3_Model
from deepd3.training.stream import DataGeneratorStream

from tensorflow.keras.callbacks import (
    ModelCheckpoint, CSVLogger,
    LearningRateScheduler)

assert tf.config.list_physical_devices('GPU')
assert tf.test.is_built_with_cuda()


def schedule(
        epoch,
        learning_rate):

    if epoch < 15:
        return learning_rate
    else:
        return learning_rate * tf.math.exp(-0.1)


sm.set_framework("tf.keras")

gpus = tf.config.experimental.list_physical_devices('GPU')
print(gpus)

# %% Randomly create train and validation sets


def split_train_val(
        file_list,
        val_ratio=0.2,
        seed=None):

    if seed is not None:
        random.seed(seed)

    shuffled_files = file_list.copy()
    random.shuffle(shuffled_files)

    split_idx = int(len(shuffled_files) * (1 - val_ratio))

    train_files = shuffled_files[:split_idx]
    val_files = shuffled_files[split_idx:]

    return train_files, val_files


def process_and_save(
        file_list,
        save_fn):

    stacks = {}
    dendrites = {}
    spines = {}
    meta = pd.DataFrame()

    # Process each file in the list
    for i, fn in enumerate(file_list):
        d = fl.load(fn)

        stacks[f"x{i}"] = d['data']['stack']
        dendrites[f"x{i}"] = d['data']['dendrite']
        spines[f"x{i}"] = d['data']['spines']

        m = pd.DataFrame([d['meta']])
        meta = pd.concat((meta, m), axis=0, ignore_index=True)

    # Save the processed data
    fl.save(save_fn,
            dict(data=dict(stacks=stacks, dendrites=dendrites, spines=spines),
                 meta=meta),
            compression='blosc')


# All the segmented data stored in a folder as d3data
d3data_file_list = list(
    Path(r"Y:\SynEmo\TrainingDataset\ongoing_dataset\d3data").iterdir())

train_files, val_files = split_train_val(
    d3data_file_list, val_ratio=0.2, seed=42)

training_data_path = Path(
    r"Y:\SynEmo\TrainingDataset\ongoing_dataset\240909_train.d3set")
process_and_save(train_files, training_data_path)

validation_data_path = Path(
    r"Y:\SynEmo\TrainingDataset\ongoing_dataset\240909_val.d3set")
process_and_save(val_files, validation_data_path)


# %% Prepare data


model_name = datetime.now().strftime("%y%m%d")
model_filename = f"model_{model_name}_2.h5"
model_output_path = Path('models', model_filename)

logger_filename = Path('models', f'model_{model_name}_logger.csv')

# Convert stack data to uint16
for d3set in [training_data_path, validation_data_path]:
    dataset = fl.load(d3set)

    for k, v in dataset['data']['stacks'].items():
        dataset['data']['stacks'][k] -= v.min()
        dataset['data']['stacks'][k] = (
            dataset['data']['stacks'][k].astype(np.uint16))

    fl.save(d3set,
            dict(data=dataset['data'],
                 meta=dataset['meta']))

# %% Generate Stream

# The "size" parameter must indicate a fraction of the original image's size
# and must be square and divisible by 8. Default (1, 128, 128).

dg_training = DataGeneratorStream(
    training_data_path,
    batch_size=32,
    target_resolution=None,
    min_content=50)

dg_validation = DataGeneratorStream(
    validation_data_path,
    batch_size=32,
    target_resolution=None,
    min_content=50,
    augment=False,
    shuffle=False)

# %% Visualize data

X, Y = dg_training[0]
i = 0

fig, ax = plt.subplots(1, 3, figsize=(12, 4))

# Plot for the first image
im0 = ax[0].imshow(X[i].squeeze(), cmap='gray')
fig.colorbar(im0, ax=ax[0])

# Plot for the second image
im1 = ax[1].imshow(Y[0][i].squeeze(), cmap='gray')
fig.colorbar(im1, ax=ax[1])

# Plot for the third image
im2 = ax[2].imshow(Y[1][i].squeeze(), cmap='gray')
fig.colorbar(im2, ax=ax[2])


# %%Create and train the model

# Create a naive DeepD3 model with a given base filter count
model = DeepD3_Model(
    filters=32,
    input_shape=(None, None, 1))

model.compile(
    Adam(learning_rate=0.0005),
    [sm.losses.dice_loss, "mse"],
    metrics=['acc', sm.metrics.iou_score])

model.summary()

epochs = 30

# Save best model automatically during training
mc = ModelCheckpoint(
    model_filename,
    save_best_only=True)

# Save metrics
csv = CSVLogger(logger_filename)

# Adjust learning rate during training to allow for better convergence
lrs = LearningRateScheduler(schedule)

start_time = time.time()
print(start_time)

# Actually train the network
history = model.fit(
    dg_training,
    batch_size=32,
    epochs=epochs,
    validation_data=dg_validation,
    callbacks=[mc, csv, lrs],
    verbose=1)

end_time = time.time()

print(f'Total time: {(end_time - start_time) / 60} minutes')

model.save(model_output_path)

# %% Plot performance

history_df = pd.read_csv(logger_filename)

for metric in ['loss', 'acc']:
    for structure in ['spines', 'dendrites']:
        # Plot value
        fig, ax = plt.subplots()
        ax.set_title(f'{structure} {metric}')
        ax.plot(history_df['epoch'],
                history_df[f'{structure}_{metric}'])
        ax.plot(history_df['epoch'],
                history_df[f'val_{structure}_{metric}'])
        ax.set_ylabel(f'{metric}')
        ax.set_xlabel('Epochs')
        ax.legend(['Training', 'Validation'])

# DeepD3 model with input shape = (None, None, 1)
# dg_training and dg_validation size should be (32, 32, 1).
# if size = (128, 128, 1), it enters infinite loop immediately.
# if size = (64, 64, 1), it starts a training, it gets stuck at first epoch

# Attempt with stack with actual size 128 x 128 (1, 128, 128)
# Still infinite loop, with both 8 and 32 filters
# by setting size as (1, 64, 64) in the generatorStream,
