import csv
import numpy as np
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
metrics = ["avg_score", "score_distribution"]

with open('rating2.csv', 'r') as f:
    f_csv = csv.reader(f, delimiter=';')
    for row in f_csv:
        data.append(row)

ratings = {}
threshold = 200

for trace in data[1:]:
    video_id, mode, rating = int(trace[2]), int(trace[3]), int(trace[5])
    if video_id not in ratings.keys():
        ratings[video_id] = [[], []]
    ratings[video_id][mode].append(rating)

x, y = [], []
index = np.arange(5) + 1
# print(index)

score_count_free = np.zeros(5)
score_count_guidance = np.zeros(5)

all_rating_free, all_rating_hybrid = 0, 0

for video_id in ratings:
    all_ratings = ratings[video_id]
    if len(all_ratings[0]) == 0 or len(all_ratings[1]) == 0:
        continue
    if metrics[args.anamode] == "avg_score":
        all_ratings[0].sort(reverse=True)
        all_ratings[1].sort(reverse=True)
        num_users = min(threshold, len(all_ratings[0]))
        avg_score_free = sum(all_ratings[0][:num_users]) / num_users
        num_users = min(threshold, len(all_ratings[1]))
        avg_score_guidance = sum(all_ratings[1][:num_users]) / num_users
        print(video_id, num_users, avg_score_free, avg_score_guidance)
        # if avg_score_free > avg_score_guidance:
        #     continue
        x.append(avg_score_free)
        y.append(avg_score_guidance)
    elif metrics[args.anamode] == "score_distribution":
        # num_users = min(threshold, len(all_ratings[0]))
        # avg_score_free = sum(all_ratings[0][:num_users]) / num_users
        # num_users = min(threshold, len(all_ratings[1]))
        # avg_score_guidance = sum(all_ratings[1][:num_users]) / num_users
        # print(video_id, num_users, avg_score_free, avg_score_guidance)
        # if avg_score_free > avg_score_guidance:
        #     continue
        # print(len(all_ratings[0]))
        all_rating_free += len(all_ratings[0])
        all_rating_hybrid += len(all_ratings[1])
        for rate in all_ratings[0]:
            score_count_free[rate - 1] += 1
        for rate in all_ratings[1]:
            score_count_guidance[rate - 1] += 1


if metrics[args.anamode] == "avg_score":
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    plt.scatter(x, y, s=80, c='lightcoral')
    plt.plot(index, index)

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.xlim((1, 5))
    plt.ylim((1, 5))
    plt.xticks(np.linspace(1, 5, 5), size=32)
    plt.yticks(np.linspace(1, 5, 5), size=32)

    ax.set_ylabel('Hybrid MOS', size=40)
    ax.set_xlabel('Free MOS', size=40)

    plt.grid()
    fig.savefig('figures/avg.pdf', bbox_inches='tight', format='pdf')
elif metrics[args.anamode] == "score_distribution":
    print(score_count_free, score_count_guidance)
    fig = plt.figure(figsize=(10, 7))
    ax = fig.add_subplot(111)
    plt.bar(index - 0.2, score_count_free / all_rating_free, width=0.4, color='lightcoral', label='free')
    plt.bar(index + 0.2, score_count_guidance / all_rating_hybrid, width=0.4, color='cyan', label='hybrid')

    ax.spines['bottom'].set_linewidth(3)
    ax.spines['top'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['right'].set_linewidth(3)
    plt.xlim((0, 6))
    plt.ylim((0, 0.4))
    plt.xticks(np.linspace(1, 5, 5), size=32)
    plt.yticks(np.linspace(0, 0.4, 5), size=32)

    ax.set_ylabel('Percentage', size=40)
    ax.set_xlabel('Score', size=40)

    plt.grid()
    plt.legend()
    fig.savefig('figures/distribution.pdf', bbox_inches='tight', format='pdf')



