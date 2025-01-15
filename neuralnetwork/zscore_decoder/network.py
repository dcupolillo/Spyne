""" Created on Wed Oct 16 10:58:40 2024
    @author: dcupolillo """

import torch
import torch.nn as nn
import torch.nn.functional as F


class ZScoreNN(nn.Module):

    def __init__(
            self,
            trace_length: int = 50,
            input_channels: int = 1,  # Input: 1 channel (the zscore trace)
            out_channels_conv1: int = 16,
            out_channels_conv2: int = 32,
            kernel_size_conv1: int = 3,
            kernel_size_conv2: int = 3,
            kernel_size_pool: int = 2,
            fc1_out_features: int = 64,
            fc2_out_features: int = 32,
            dropout: float = 0.5
    ) -> None:

        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_channels,
            out_channels=out_channels_conv1,
            kernel_size=kernel_size_conv1,
            padding=1
        )
        self.bn1 = nn.BatchNorm1d(out_channels_conv1)

        self.conv2 = nn.Conv1d(
            in_channels=out_channels_conv1,
            out_channels=out_channels_conv2,
            kernel_size=kernel_size_conv2,
            padding=1
        )
        self.bn2 = nn.BatchNorm1d(out_channels_conv2)

        self.pool = nn.MaxPool1d(kernel_size=kernel_size_pool)

        self.fc1 = nn.Linear(
            int(out_channels_conv2 * (trace_length / kernel_size_pool)),
            fc1_out_features)
        self.fc2 = nn.Linear(fc1_out_features, fc2_out_features)
        self.fc3 = nn.Linear(fc2_out_features, 1)

        self.dropout = nn.Dropout(dropout)

    def forward(self, x):

        x = self.conv1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten
        x = x.view(x.size(0), -1)

        # Fully Connected Layers
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)

        x = self.fc3(x)
        x = torch.sigmoid(x)

        return x
