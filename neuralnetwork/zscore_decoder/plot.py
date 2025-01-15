""" Created on Wed Oct 16 11:56:28 2024
    @author: dcupolillo """

import matplotlib.pyplot as plt
import torch
import numpy as np
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, accuracy_score, precision_score, recall_score, f1_score)


def to_list(tensor_list):
    if isinstance(tensor_list[0], torch.Tensor):
        return [t.cpu().item() for t in tensor_list]
    return tensor_list


def plot_loss_accuracy(
    train_loss,
    train_acc,
    validation_loss,
    validation_acc,
    train_f1,
    validation_f1,
    loss_ylim=None,
    xlim=None,
    fontsize=None,
    train_color="blue",
    valid_color="green"
):
    """
    Code to plot loss and accuracy

    Args:
      train_loss: list
        Log of training loss
      validation_loss: list
        Log of validation loss
      train_acc: list
        Log of training accuracy
      validation_acc: list
        Log of validation accuracy

    Returns:
      Nothing
    """
    fontsize = fontsize if fontsize is not None else 10

    # Ensure inputs are lists
    train_loss = to_list(train_loss)
    validation_loss = to_list(validation_loss)
    train_acc = to_list(train_acc)
    validation_acc = to_list(validation_acc)
    train_f1 = to_list(train_f1)
    validation_f1 = to_list(validation_f1)

    epochs = len(train_loss)

    if xlim is None:
        xlim = (0, epochs)

    epochs = len(train_loss)
    fig, axes = plt.subplots(
        1, 3,
        figsize=(13, 4),
        layout="constrained")

    ax1, ax2, ax3 = axes

    ax1.plot(
        list(range(epochs)),
        train_loss,
        color=train_color,
        label='Training Loss',
        clip_on=False)
    ax1.plot(
        list(range(epochs)),
        validation_loss,
        color=valid_color,
        label='Validation Loss',
        clip_on=False)
    ax1.set_ylabel('Loss', fontsize=fontsize)
    if loss_ylim:
        ax1.set_ylim(loss_ylim)

    ax2.plot(
        list(range(epochs)),
        train_acc,
        color=train_color,
        label='Training Acc.',
        clip_on=False)
    ax2.plot(
        list(range(epochs)),
        validation_acc,
        color=valid_color,
        label='Validation Acc.',
        clip_on=False)
    ax2.set_ylabel('Acc.', fontsize=fontsize)
    ax2.set_ylim(0, 100)

    ax3.plot(
        range(epochs),
        train_f1,
        color=train_color,
        label='Training F1 Score',
        clip_on=False)
    ax3.plot(
        range(epochs),
        validation_f1,
        color=valid_color,
        label='Validation F1 Score',
        clip_on=False)
    ax3.set_ylabel('F1 Score', fontsize=fontsize)
    ax3.set_xlabel('Epochs', fontsize=fontsize)
    ax3.set_ylim(0, 1.)

    for ax in axes:
        ax.set_xlim(xlim)
        ax.legend(fontsize=fontsize)
        ax.set_xlabel('Epochs', fontsize=fontsize)
        ax.tick_params("both", labelsize=fontsize)
        ax.spines[["top", "right"]].set_visible(False)
        ax.spines[["bottom", "left"]].set_position(("outward", 20))


def get_predictions_and_labels(
        model,
        data_loader,
        device,
        threshold=0.5
):

    model.eval()
    y_true, y_pred = [], []

    with torch.no_grad():
        for data, target in data_loader:
            data = data.float().unsqueeze(1).to(device)
            target = target.float().to(device)

            # Forward pass
            output = model.forward(data)
            predicted = (output > threshold).float()

            y_true.extend(target.cpu().numpy().flatten())
            y_pred.extend(predicted.cpu().numpy().flatten())

    return y_true, y_pred


def plot_confusion_matrix(
        y_true,
        y_pred,
        classes=None,
        fontsize=10,
        cmap="Blues"):

    # Compute confusion matrix
    raw_cm = confusion_matrix(y_true, y_pred)
    cm_percentage = (
        raw_cm.astype('float') / raw_cm.sum(axis=1, keepdims=True))

    annot = []
    for i in range(raw_cm.shape[0]):
        row = []
        for j in range(raw_cm.shape[1]):
            percentage = f"{cm_percentage[i, j]:.2%}"
            raw = f"({raw_cm[i, j]})"
            row.append(f"{percentage}\n{raw}")
        annot.append(row)

    if classes is None:
        classes = ["Class 0", "Class 1"]

    # Plot the confusion matrix
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(
        cm_percentage,
        ax=ax,
        annot=annot,
        fmt="",
        cmap=cmap,
        xticklabels=classes,
        yticklabels=classes,
        cbar=False,
        annot_kws={"size": fontsize}
    )
    ax.set_xlabel("Predicted", fontsize=fontsize)
    ax.set_ylabel("True", fontsize=fontsize)
    ax.tick_params('both', labelsize=fontsize)
