""" Created on Wed Oct 16 11:44:38 2024
    @author: dcupolillo """

from pathlib import Path
import torch
from torchsummary import summary
import flammkuchen as fl
from torch import nn

from spyne.neuralnetwork.zscore_decoder.zscoredataset import ZScoreDataset
from spyne.neuralnetwork.zscore_decoder.network import ZScoreNN
# from spyne.neuralnetwork.zscore_decoder.LSTM_network import LSTMClassifier
from spyne.neuralnetwork.zscore_decoder.utils import (
    split, train, set_device,
    set_seed, compute_class_weights)
from spyne.neuralnetwork.zscore_decoder.plot import (
    plot_loss_accuracy, plot_confusion_matrix, get_predictions_and_labels)

device = set_device()
model_output_path = Path("zscore_new_best_model.pth")


# %%

data = fl.load(
    r"neuralnetwork/zscore_decoder/datasets/bigger_zscore_labels.h5")

set_seed(44)
batch_size = 32
validation_split = 0.5
test_split = 0.25
epochs = 100
learning_rate = 1e-3
lambda1 = 1e-4
lambda2 = 1e-2
patience = 20
dropout = 0.1
augment_probability = 0.6
noise_level = 0.4

dataset = ZScoreDataset(
    data,
    event_range=None,
    augment=True,
    augment_probability=augment_probability,
    noise_level=noise_level)

labels = torch.tensor([int(dataset[i][1].item()) for i in range(len(dataset))])
pos_weight = compute_class_weights(labels).to(device)
criterion = nn.BCELoss()

train_loader, valid_loader, test_loader = split(
    dataset,
    validation_split=validation_split,
    test_split=test_split,
    batch_size=batch_size)

zscore_classifier = ZScoreNN(
    trace_length=len(dataset[0][0]),
    dropout=dropout
).to(device)

summary(
    zscore_classifier,
    input_size=(1, len(dataset[0][0])),
    device=device)

# Training
(
     model,
     train_loss,
     validation_loss,
     train_acc,
     validation_acc,
     train_f1,
     validation_f1
) = train(
    train_loader=train_loader,
    valid_loader=valid_loader,
    model=zscore_classifier,
    criterion=criterion,
    device=device,
    epochs=epochs,
    lr=learning_rate,
    lambda1=lambda1,
    lambda2=lambda2,
    patience=patience)

# Inference on test dataset
labels, predictions = get_predictions_and_labels(
    model,
    test_loader,
    device)

# Plotting resuts
plot_loss_accuracy(
    train_loss,
    train_acc,
    validation_loss,
    validation_acc,
    train_f1,
    validation_f1,
    loss_ylim=(0, 20),
    xlim=(0, 25),
    fontsize=14)

plot_confusion_matrix(
    labels,
    predictions,
    classes=["No Event", "Event"],
    fontsize=14)


# Optionally save model and results
# torch.save(
#     model.state_dict(),
#     model_output_path)
