import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from parser import args
import os
import glob
import random

random.seed(1)
torch.manual_seed(1)


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


def load_data(input_dir=args.following_data, skips=args.skips, history_window=args.history_window):
    train_data, test_data = [], []
    input_files = sorted(glob.glob(input_dir + os.sep + '*'))
    label_counts = np.zeros(args.output_dim)
    # users = []
    # for input_file in input_files:
    #     suffix = input_file.split('/')[-1].split('.')[0]
    #     user = suffix[:4]
    #     if user not in users:
    #         users.append(user)
    # random.shuffle(users)
    random.shuffle(input_files)
    training_size = round(len(input_files) * 0.8)
    # print(training_size, len(input_files) - training_size)
    file_idx = 0
    for input_file in input_files:
        # suffix = input_file.split('/')[-1].split('.')[0]
        # user = suffix[:4]
        # video_id = int(suffix.split('_')[-1])
        file_idx = file_idx + 1
        data = np.load(input_file)
        seq_len = data.shape[0]
        for i in range(1, seq_len - skips - 1):
            # input = torch.from_numpy(data[:i, :args.input_dim]).float()
            input = data[i - 1: i, : args.input_dim]
            for j in range(2, history_window + 1):
                added_input = data[max(0, i - j): max(0, i - j) + 1, : args.input_dim]
                input = np.concatenate((input, added_input), axis=0)
            input = torch.from_numpy(input).float()
            label = torch.from_numpy(data[i: i + 1, 2]).long()
            for j in range(i + 1, i + skips + 1):
                if label.item() == 0:
                    break
                sample = torch.from_numpy(data[j: j + 1, 2]).long()
                label = torch.mul(label, sample)
            if file_idx <= training_size:
                train_data.append((input, label))
            else:
                test_data.append((input, label))
            # if video_id % 5 == 0:
            #     test_data.append((input, label))
            # else:
            #     train_data.append((input, label))
            label_counts[label.item()] += 1
    print(label_counts)
    return train_data, test_data


def save_logs(training_logs, test_logs):
    training_logs = np.array(training_logs)
    test_logs = np.array(test_logs)
    template = args.log_dir + os.sep + '{0}_{1}_lr={2}_{3}.npy'
    np.save(template.format('train', args.skips, args.lr, 'following'), training_logs)
    np.save(template.format('test', args.skips, args.lr, 'following'), test_logs)


def training(train_data, test_data, num_epochs=args.num_epochs):
    # model = LSTM(args.input_dim, args.hidden_dim, args.output_dim)
    # model = FC(args.history_window * args.input_dim, args.output_dim)
    model = CNN(args.input_dim, args.hidden_dim, args.output_dim)
    loss_function = nn.CrossEntropyLoss()
    training_logs, test_logs = [], []
    best_model, min_loss, best_precision = None, 1.0, 0.0
    # loss_function = nn.MSELoss()
    # optimizer = optim.Adam(model.parameters(), lr=args.lr)
    optimizer = optim.SGD(model.parameters(), lr=args.lr)
    model_path_template = args.model_dir + os.sep + "{0}_lookahead={1}.pth"
    model_path = model_path_template.format("following", args.skips + 1)
    for _ in range(num_epochs):
        model.train()
        losses = []
        random.shuffle(train_data)
        true_follows, true_unfollows, false_unfollows, false_follows = 0, 0, 0, 0
        for (input, label) in train_data:
            model.zero_grad()
            output = model(input)
            prob = F.softmax(output, dim=1)
            loss = loss_function(output, label)
            losses.append(loss.item())
            loss.backward()
            optimizer.step()
            prob_follow = prob[0][1].item()
            if label.item() == 0:
                if prob_follow < args.threshold:
                    true_unfollows += 1
                else:
                    false_follows += 1
            else:
                if prob_follow > args.threshold:
                    true_follows += 1
                else:
                    false_unfollows += 1
        train_loss = round(sum(losses) / len(losses), 4)
        recall_follows = round(true_follows / (true_follows + false_unfollows), 4)
        recall_unfollows = round(true_unfollows / (true_unfollows + false_follows), 4)
        acc = true_follows + true_unfollows
        tot = acc + false_follows + false_unfollows
        train_precision = round(acc / tot, 4)
        message_template = "train round {0}: loss {1}; recall {2} {3}; precision {4}"
        print(message_template.format(_, train_loss, recall_follows, recall_unfollows,
                                      train_precision))
        training_logs.append([train_loss, recall_follows, recall_unfollows, train_precision])

        losses = []
        true_follows, true_unfollows, false_unfollows, false_follows = 0, 0, 0, 0
        with torch.no_grad():
            for (input, label) in test_data:
                output = model(input)
                prob = F.softmax(output, dim=1)
                loss = loss_function(output, label)
                losses.append(loss.item())
                prob_follow = prob[0][1].item()
                if label.item() == 0:
                    if prob_follow < args.threshold:
                        true_unfollows += 1
                    else:
                        false_follows += 1
                else:
                    if prob_follow > args.threshold:
                        true_follows += 1
                    else:
                        false_unfollows += 1
        test_loss = round(sum(losses) / len(losses), 4)
        recall_follows = round(true_follows / (true_follows + false_unfollows), 4)
        recall_unfollows = round(true_unfollows / (true_unfollows + false_follows), 4)
        acc = true_follows + true_unfollows
        tot = acc + false_follows + false_unfollows
        test_precision = round(acc / tot, 4)
        if test_loss < min_loss:
            min_loss = test_loss
            best_precision = test_precision
            torch.save(model.state_dict(), model_path)
        message_template = "test round {0}: loss {1} ({2}); recall {3} {4}; precision {5} ({6})"
        print(message_template.format(_, test_loss, min_loss, recall_follows, recall_unfollows,
                                      test_precision, best_precision))
        test_logs.append([test_loss, recall_follows, recall_unfollows, test_precision])
    return training_logs, test_logs


if __name__ == '__main__':
    train_data, test_data = load_data()
    training_logs, test_logs = training(train_data, test_data)
    save_logs(training_logs, test_logs)
