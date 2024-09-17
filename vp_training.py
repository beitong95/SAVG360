import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from parser import args
import os
import glob
import random
from utils import *

random.seed(1)
torch.manual_seed(1)


class LR(nn.Module):

    def __init__(self, history_window=args.history_window * 5, output_dim=1):
        super(LR, self).__init__()
        self.linear = nn.Linear(history_window, output_dim)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


def load_data(input_dir=args.vp_data, skips=args.skips * 5, history_window=args.history_window * 5):
    train_data, test_data = [], []
    input_files = sorted(glob.glob(input_dir + os.sep + '*'))
    random.shuffle(input_files)
    training_size = round(len(input_files) * 0.8)
    file_idx = 0
    for input_file in input_files:
        file_idx = file_idx + 1
        data = np.load(input_file)
        seq_len = data.shape[0]
        for i in range(1, seq_len - skips - 1):
            # input = torch.from_numpy(data[:i, :args.input_dim]).float()
            input = data[i - 1: i, :]
            for j in range(2, history_window + 1):
                added_input = data[max(0, i - j): max(0, i - j) + 1, :]
                input = np.concatenate((input, added_input), axis=0)
            input = torch.from_numpy(input).float()
            label = torch.from_numpy(data[i + skips: i + skips + 1, :]).float()
            if file_idx <= training_size:
                train_data.append((input, label))
            else:
                test_data.append((input, label))
    print(len(train_data), len(test_data))
    return train_data, test_data


def lonlat2tile(theta, phi):
    x = round((0.5 - phi) * args.tile_h - 0.5)
    y = round((0.5 + theta) * args.tile_w - 0.5)
    return (x, y)


def lonlat2dist(theta0, phi0, theta1, phi1):
    xyz0 = lonlat2xyz(theta0 * 360, phi0 * 180)
    xyz1 = lonlat2xyz(theta1 * 360, phi1 * 180)
    dist = np.sum((xyz0 - xyz1) ** 2)
    return dist


def save_logs(training_logs, test_logs):
    training_logs = np.array(training_logs)
    test_logs = np.array(test_logs)
    template = args.log_dir + os.sep + '{0}_{1}_lr={2}_{3}.npy'
    np.save(template.format('train', args.skips, args.lr, 'vp'), training_logs)
    np.save(template.format('test', args.skips, args.lr, 'vp'), test_logs)


def training(train_data, num_epochs=args.num_epochs):
    x_model = LR()
    y_model = LR()
    loss_function = nn.MSELoss()
    training_logs, test_logs = [], []
    best_model, min_error, best_precision = None, 1.0, 0.0
    x_optimizer = optim.SGD(x_model.parameters(), lr=args.lr)
    y_optimizer = optim.SGD(y_model.parameters(), lr=args.lr)
    model_path_template = args.model_dir + os.sep + "{0}_{1}_lookahead={2}.pth"
    for _ in range(num_epochs):
        x_model.train()
        y_model.train()
        losses = []
        errors = []
        random.shuffle(train_data)
        acc, tot = 0, 0
        for (input, label) in train_data:
            x_model.zero_grad()
            x_output = x_model(input[:,0])
            x_loss = loss_function(x_output, label[0][0])
            losses.append(x_loss.item())
            x_loss.backward()
            x_optimizer.step()

            y_model.zero_grad()
            y_output = y_model(input[:, 1])
            y_loss = loss_function(y_output, label[0][1])
            losses.append(y_loss.item())
            y_loss.backward()
            y_optimizer.step()

            error = lonlat2dist(x_output.item(), y_output.item(),
                           label[0][0].item(), label[0][1].item())
            errors.append(error)
            if error <= 0.1:
                acc += 1
            tot += 1

        train_loss = round(sum(losses) / len(losses), 6)
        avg_error = round(sum(errors) / len(errors), 6)
        train_precision = round(acc / tot, 4)
        message_template = "train round {0}: loss {1}; avg error {2}, precision {3}"
        print(message_template.format(_, train_loss, avg_error, train_precision))
        training_logs.append([train_loss, avg_error, train_precision])

        losses = []
        acc, tot = 0, 0
        with torch.no_grad():
            for (input, label) in test_data:
                x_output = x_model(input[:, 0])
                x_loss = loss_function(x_output, label[0][0])
                losses.append(x_loss.item())

                y_output = y_model(input[:, 1])
                y_loss = loss_function(y_output, label[0][1])
                losses.append(y_loss.item())

                error = lonlat2dist(x_output.item(), y_output.item(),
                                    label[0][0].item(), label[0][1].item())
                errors.append(error)
                if error <= 0.1:
                    acc += 1
                tot += 1

        test_loss = round(sum(losses) / len(losses), 6)
        avg_error = round(sum(errors) / len(errors), 6)
        test_precision = round(acc / tot, 4)
        if avg_error < min_error:
            min_error = avg_error
            best_precision = test_precision
            torch.save(x_model.state_dict(), model_path_template.format("vp", "x", args.skips + 1))
            torch.save(y_model.state_dict(), model_path_template.format("vp", "y", args.skips + 1))
        message_template = "test round {0}: loss {1}; avg_error {2} ({3}); precision {4} ({5})"
        print(message_template.format(_, test_loss, avg_error, min_error, test_precision, best_precision))
        test_logs.append([test_loss, test_precision])


    return training_logs, test_logs


if __name__ == '__main__':
    train_data, test_data = load_data()
    training_logs, test_logs = training(train_data)
    save_logs(training_logs, test_logs)
