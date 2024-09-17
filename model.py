import torch
import torch.nn as nn
from parser import args


class LSTM(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, dropout=args.dropout):
        super(LSTM, self).__init__()
        self.hidden_dim = hidden_dim
        self.lstm = nn.LSTM(input_dim, hidden_dim, dropout=dropout)
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        lstm_out, _ = self.lstm(input.view(len(input), 1, -1))
        # print(lstm_out[-1].size())
        # hidden_out = self.relu(lstm_out.view(len(input), -1))
        output = self.fc(lstm_out[-1])
        return output


class CNN(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, kernel=2, stride=1):
        super(CNN, self).__init__()
        self.hidden_dim = hidden_dim
        self.conv = nn.Conv1d(input_dim, hidden_dim, kernel, stride)
        self.pooling = nn.MaxPool1d(args.history_window - (kernel - 1))
        self.relu = nn.ReLU()
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, input):
        # print(input.size())
        x = input.unsqueeze(dim=0).permute(0, 2, 1)
        x = self.pooling(self.relu(self.conv(x)))
        x = torch.flatten(x, 1)
        output = self.fc(x)
        return output


class FC(nn.Module):

    def __init__(self, input_dim, output_dim):
        super(FC, self).__init__()
        self.fc = nn.Linear(input_dim, output_dim)
        self.relu = nn.ReLU()

    def forward(self, input):
        output = self.fc(input.view(1, -1))
        return output


class LR(nn.Module):

    def __init__(self, history_window=args.history_window * 5, output_dim=1):
        super(LR, self).__init__()
        self.linear = nn.Linear(history_window, output_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred