import csv
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import rcParams
from parser import args

data = []
config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 20,
}
rcParams.update(config)
mode = "loss"


training_logs = []
test_logs = []
template = args.log_dir + os.sep + '{0}_{1}_lr={2}_following.npy'
for skips in range(3):
    training_log = np.load(template.format('train', skips + 1, args.lr))
    test_log = np.load(template.format('test', skips + 1, args.lr))
    training_logs.append(training_log)
    test_logs.append(test_log)

training_logs = np.array(training_logs)
test_logs = np.array(test_logs)

if mode == "loss":
    data = training_logs[:, :, 0]
    # data = test_logs[:, :, 0]
    print(data.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for skips in range(3):
        x = np.linspace(1, 100, 100)
        y = data[skips]
        # for i in range(1, 100):
            # y[i] = min(y[i-1], y[i])
        label = 'look ahead = ' + str(skips + 2)
        plt.plot(x, y, label=label)
    # plt.bar(index - 0.2, score_count_free, width=0.4, color='lightcoral', label='free')
    # plt.bar(index + 0.2, score_count_guidance, width=0.4, color='cyan', label='guidance')

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.xlim((0, 100))
    plt.ylim((0.3, 0.5))
    plt.xticks(np.linspace(0, 100, 5), size=32)
    plt.yticks(np.linspace(0.3, 0.5, 5), size=32)

    ax.set_ylabel('Loss', size=40)
    ax.set_xlabel('Training Round', size=40)

    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig('figures/loss.pdf', bbox_inches='tight', format='pdf')

if mode == "recall":
    data = training_logs[:, :, 1]
    # data = test_logs[:, :, 1]
    print(data.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for skips in range(3):
        x = np.linspace(1, 100, 100)
        y = data[skips]
        for i in range(1, 100):
            y[i] = max(y[i-1], y[i])
        label = 'look ahead = ' + str(skips + 2)
        plt.plot(x, y, label=label)
    # plt.bar(index - 0.2, score_count_free, width=0.4, color='lightcoral', label='free')
    # plt.bar(index + 0.2, score_count_guidance, width=0.4, color='cyan', label='guidance')

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.xlim((0, 100))
    plt.ylim((0.75, 1.0))
    plt.xticks(np.linspace(0, 100, 5), size=32)
    plt.yticks(np.linspace(0.75, 1.0, 6), size=32)

    ax.set_ylabel('Recall', size=40)
    ax.set_xlabel('Training Round', size=40)

    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig('figures/recall.pdf', bbox_inches='tight', format='pdf')

if mode == "precision":
    data = training_logs[:, :, 3]
    # data = test_logs[:, :, 3]
    print(data.shape)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    for skips in range(3):
        x = np.linspace(1, 100, 100)
        y = data[skips]
        for i in range(1, 100):
            y[i] = max(y[i-1], y[i])
        label = 'look ahead = ' + str(skips + 2)
        plt.plot(x, y, label=label)
    # plt.bar(index - 0.2, score_count_free, width=0.4, color='lightcoral', label='free')
    # plt.bar(index + 0.2, score_count_guidance, width=0.4, color='cyan', label='guidance')

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.xlim((0, 100))
    plt.ylim((0.6, 1.0))
    plt.xticks(np.linspace(0, 100, 5), size=32)
    plt.yticks(np.linspace(0.6, 1.0, 6), size=32)

    ax.set_ylabel('Precision', size=40)
    ax.set_xlabel('Training Round', size=40)

    plt.grid()
    plt.legend()
    plt.show()
    fig.savefig('figures/precision.pdf', bbox_inches='tight', format='pdf')


