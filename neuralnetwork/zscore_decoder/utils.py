""" Created on Wed Oct 16 11:36:39 2024
    @author: dcupolillo """

import numpy as np
import torch
from torch import nn
import time
from torch.utils.data import (
    DataLoader, SubsetRandomSampler, WeightedRandomSampler)
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.metrics import f1_score
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
import random
import copy


def set_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def set_seed(
        seed=None,
        seed_torch=True
):

    if seed is None:
        seed = np.random.choice(2 ** 32)
    random.seed(seed)
    np.random.seed(seed)

    if seed_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

    print(f'Random seed {seed} has been set.')


def compute_class_weights(labels):
    """
    Compute class weights for handling class imbalance.
    Args:
    labels: List or tensor of all labels in the dataset.

    Returns:
    weights: Tensor of class weights for each sample.
    """
    class_counts = torch.bincount(labels)
    total_samples = len(labels)
    epsilon = 1e-9
    class_weights = total_samples / (class_counts + epsilon)
    pos_weight = class_weights[1] / class_weights[0]

    return torch.tensor([pos_weight])


def split(
        dataset,
        validation_split=0.3,
        test_split=0.15,
        batch_size=32
) -> tuple:

    dataset_size = len(dataset)
    indices = list(range(dataset_size))
    labels = torch.tensor(
        [int(dataset[i][1].item()) for i in indices], dtype=torch.long)

    # Compute weights
    class_counts = torch.bincount(labels)
    class_weights = 1.0 / class_counts
    sample_weights = torch.tensor(
        [class_weights[label] for label in labels])

    strat_split = StratifiedShuffleSplit(
        n_splits=1,
        test_size=test_split,
        train_size=1 - test_split - validation_split
    )
    train_indices, temp_indices = next(strat_split.split(indices, labels))

    strat_split_val_test = StratifiedShuffleSplit(
        n_splits=1,
        test_size=validation_split / (
            validation_split + (1 - test_split - validation_split))
    )
    val_indices, test_indices = next(
        strat_split_val_test.split(
            temp_indices, [labels[i] for i in temp_indices]))

    # Create weighted sample weights for each split
    train_weights = [sample_weights[i] for i in train_indices]
    val_weights = [sample_weights[i] for i in val_indices]
    test_weights = [sample_weights[i] for i in test_indices]

    # train_sampler = WeightedRandomSampler(
    #     train_weights, num_samples=len(train_indices), replacement=True)
    # val_sampler = WeightedRandomSampler(
    #     val_weights, num_samples=len(val_indices), replacement=True)
    # test_sampler = WeightedRandomSampler(
    #     test_weights, num_samples=len(test_indices), replacement=True)

    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)
    test_sampler = SubsetRandomSampler(test_indices)

    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler
    )
    valid_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler
    )
    test_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=test_sampler
    )

    return train_loader, valid_loader, test_loader


def compute_l1_regularization(model):
    l1_norm = sum(p.abs().sum() for p in model.parameters())
    return l1_norm


def train(
        train_loader,
        valid_loader,
        model,
        criterion,
        device,
        epochs,
        lr,
        lambda1,
        lambda2,
        patience,
):
    """
    Training loop

    Args:
    model: nn.module
      Neural network instance
    device: string
      GPU/CUDA if available, CPU otherwise
    epochs: int
      Number of epochs
    train_loader: torch.loader
      Training Set
    validation_loader: torch.loader
      Validation set

    Returns:
    Nothing
    """

    model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
        weight_decay=lambda2)
    scheduler = ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True)

    train_loss, validation_loss = [], []
    train_acc, validation_acc = [], []
    train_f1, validation_f1 = [], []

    best_acc = 0.0
    best_model = copy.deepcopy(model)
    wait = 0

    start_time = time.time()

    for epoch in range(epochs):

        model.train()
        correct, total = 0, 0
        running_loss = 0.0
        all_preds, all_targets = [], []

        # Looping through all the traces
        for data, target in tqdm(
                train_loader, desc=f"Training Epoch {epoch+1}/{epochs}"):

            # Change dtype before sending to device
            data = data.float()
            data = data.unsqueeze(1)
            target = target.float()
            data, target = data.to(device), target.to(device)

            optimizer.zero_grad()

            # Forward pass
            output = model.forward(data)
            loss = criterion(output, target.view(-1, 1))

            l1_norm = compute_l1_regularization(model)
            loss += lambda1 * l1_norm
            running_loss += loss

            loss.backward()  # Backpropagate
            optimizer.step()  # Update weights

            # Get accuracy
            # _, predicted = torch.max(output, dim=1)
            predicted = (output > 0.5).float()
            all_preds.extend(predicted.cpu().numpy().flatten())
            all_targets.extend(target.cpu().numpy().flatten())
            correct = torch.eq(predicted, target).sum().item()
            total += target.size(0)

        train_loss.append(running_loss)
        train_acc.append((correct/total)*100)
        train_f1.append(f1_score(all_targets, all_preds, zero_division=0))

        # Validation
        model.eval()
        correct, total = 0, 0
        running_loss = 0.0
        all_preds, all_targets = [], []

        with torch.no_grad():  # Disable gradient calculation for validation

            for data, target in valid_loader:

                data = data.float()
                data = data.unsqueeze(1)
                target = target.float()
                data, target = data.to(device), target.to(device)

                output = model.forward(data)

                loss = criterion(output, target.view(-1, 1))
                running_loss += loss

                # We will not be calculating gradient here!!!
                # _, predicted = torch.max(output, dim=1)
                predicted = (output > 0.5).float()
                all_preds.extend(predicted.cpu().numpy().flatten())
                all_targets.extend(target.cpu().numpy().flatten())
                correct = torch.eq(predicted, target).sum().item()
                total += target.size(0)

        validation_loss.append(running_loss.cpu().detach().float())
        val_acc = (correct / total) * 100
        validation_acc.append(val_acc)
        validation_f1.append(f1_score(all_targets, all_preds, zero_division=0))

        scheduler.step(running_loss)

        # Early Stopping Check based on accuracy
        if val_acc > best_acc:
            best_acc = val_acc
            best_model = copy.deepcopy(model)
            wait = 0
        else:
            wait += 1

        if wait > patience:
            print(f"Early stopping at epoch: {epoch+1}")
            break

        print("\tTrain loss:", train_loss[-1],
              "Validation loss:", validation_loss[-1])

    end_time = time.time()
    print(f"Time for training: {(end_time - start_time) / 60.0} mins.")

    return (
        best_model,
        train_loss,
        validation_loss,
        train_acc,
        validation_acc,
        train_f1,
        validation_f1)


def evaluate_model_on_test(
        model,
        test_loader,
        device,
) -> tuple:

    model.eval()

    all_outputs = []
    with torch.no_grad():
        for data, _ in test_loader:
            data = data.float().unsqueeze(1).to(device)
            output = model(data)
            all_outputs.append(output.cpu().numpy())

    all_outputs = np.concatenate(all_outputs).ravel()
    threshold = np.percentile(all_outputs, 99)

    (
     batched_data,
     batched_predictions,
     batched_labels,
     batched_raw_outputs) = [], [], [], []

    with torch.no_grad():
        for data, labels in test_loader:
            data = data.float().unsqueeze(1).to(device)
            labels = labels.float().to(device)

            # Forward pass
            output = model.forward(data)
            predicted = (output > threshold).float()

            # Store predictions and true labels grouped by batch
            batched_data.append(data.cpu().numpy())
            batched_predictions.append(predicted.cpu().numpy())
            batched_labels.append(labels.cpu().numpy())
            batched_raw_outputs.append(output.cpu().numpy())

    batched_predictions = [
        np.squeeze(arr) for arr in batched_predictions]
    batched_raw_outputs = [
        np.squeeze(arr) for arr in batched_raw_outputs]

    return (
        batched_data,
        batched_predictions,
        batched_labels,
        batched_raw_outputs)
