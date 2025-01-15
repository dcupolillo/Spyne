""" Created on Mon Dec 16 17:34:25 2024
    @author: dcupolillo """

from torch import nn


class LSTMClassifier(nn.Module):

    def __init__(
            self,
            input_size=1,
            hidden_size=64,
            num_layers=2,
            dropout=0.2
    ):

        super(LSTMClassifier, self).__init__()

        self.lstm = nn.LSTM(
            input_size,
            hidden_size,
            num_layers,
            batch_first=True,
            dropout=dropout)

        self.fc = nn.Linear(hidden_size, 1)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        lstm_out, _ = self.lstm(x)
        out = self.fc(lstm_out[:, -1, :])

        return self.sigmoid(out)
