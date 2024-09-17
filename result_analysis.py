import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams
import scipy.stats as st
from parser import args

config = {
    "mathtext.fontset": 'stix',
    "font.family": 'serif',
    "font.serif": ['Times New Roman'],
    "font.size": 20,
}
rcParams.update(config)


# ---------------- prediction accuracy ----------------
x = np.linspace(1, 3, 3)
# prediction_acc = [
#     [0.8713, 0.8355, 0.8079],
#     [0.8373, 0.7810, 0.7079],
#     [0.8859, 0.8635, 0.8509],
#     [0.8289, 0.7903, 0.7716],
# ]
# prediction_acc = [
#     [0.8425, 0.7983, 0.7648],
#     [0.8053, 0.7393, 0.6584],
#     [0.8587, 0.8287, 0.8101],
#     [0.8052, 0.7626, 0.7367],
# ]
prediction_acc = [
    [0.8196, 0.7773, 0.7404],
    [0.7703, 0.7049, 0.6227],
    [0.8551, 0.8172, 0.7951],
    [0.7935, 0.7453, 0.7166],
]

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111)
plt.bar(x - 0.3, prediction_acc[1], color='lightcoral', width=0.2, label='LR')
plt.bar(x - 0.1, prediction_acc[0], color='skyblue', width=0.2, label='LR+VMP')
plt.bar(x + 0.1, prediction_acc[3], color='orange', width=0.2, label='NG')
plt.bar(x + 0.3, prediction_acc[2], color='cyan', width=0.2, label='NG+VMP')

ax.spines['bottom'].set_linewidth(3)
ax.spines['top'].set_linewidth(3)
ax.spines['left'].set_linewidth(3)
ax.spines['right'].set_linewidth(3)
plt.ylim((0.6, 1.))
plt.xticks(np.linspace(1, 3, 3), ['Lookahead=1', 'Lookahead=2', 'Lookahead=3'],
           rotation=20, size=32)
plt.yticks(np.linspace(0.6, 1., 5), size=32)
#
ax.set_ylabel('Precision', size=40)

plt.grid()
plt.legend(ncol=4, bbox_to_anchor=(1.05, 1.15))
# fig.savefig('figures/prediction_acc_bar_tile46.pdf', bbox_inches='tight', format='pdf')
# fig.savefig('figures/prediction_acc_bar_tile24.pdf', bbox_inches='tight', format='pdf')
fig.savefig('figures/prediction_acc_bar_tile68.pdf', bbox_inches='tight', format='pdf')

# methods = ['LR+WMP', 'LR']
# colors = ['red', 'blue']
# distribution_acc = [[0., ], [0., ]]
# with open('saved_results/prediction_acc_lrwmp_' + str(args.skips * 2) , 'r') as f:
#     for line in f:
#         acc = float(line.strip())
#         distribution_acc[0].append(acc)
# with open('saved_results/prediction_acc_lr_' + str(args.skips * 2) , 'r') as f:
#     for line in f:
#         acc = float(line.strip())
#         distribution_acc[1].append(acc)
#
# distribution_acc[0].sort()
# distribution_acc[1].sort()
# num_traces = len(distribution_acc[0])
#
# y = np.linspace(0, num_traces - 1, num_traces) / (num_traces - 1)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# for i in range(len(methods)):
#     plt.plot(distribution_acc[i], y, c=colors[i], label=methods[i])
#
# ax.spines['bottom'].set_linewidth(3)
# ax.spines['top'].set_linewidth(3)
# ax.spines['left'].set_linewidth(3)
# ax.spines['right'].set_linewidth(3)
# plt.xlim((0., 1.))
# plt.ylim((0., 1.))
# plt.xticks(np.linspace(0., 1., 6), size=32)
# plt.yticks(np.linspace(0., 1., 6), size=32)
#
# ax.set_xlabel('Precision', size=40)
# ax.set_ylabel('CDF', size=40)
#
# plt.grid()
# plt.legend()
# fig.savefig('figures/prediction_acc_cdf_{0}.pdf'.format(args.skips*2), bbox_inches='tight', format='pdf')

# ---------------- reward ----------------
# methods = ['LR+VMP', 'LR']
# colors = ['red', 'blue']
# distribution_acc = [[], []]
# min_reward = 0.0
#
# with open('saved_results/lr_reward_4_6_hsdpa', 'r') as f:
#     for line in f:
#         reward = float(line.strip())
#         min_reward = min(min_reward, reward)
#         distribution_acc[0].append(reward)
# with open('saved_results/lrvmp_reward_4_6_hsdpa' , 'r') as f:
#     for line in f:
#         reward = float(line.strip())
#         min_reward = min(min_reward, reward)
#         distribution_acc[1].append(reward)
#
# distribution_acc[0].append(min_reward)
# distribution_acc[1].append(min_reward)
# distribution_acc[0].sort()
# distribution_acc[1].sort()
# num_traces = len(distribution_acc[0])
#
# y = np.linspace(0, num_traces - 1, num_traces) / (num_traces - 1)
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
# for i in range(len(methods)):
#     plt.plot(distribution_acc[i], y, c=colors[i], label=methods[i])
#
# ax.spines['bottom'].set_linewidth(3)
# ax.spines['top'].set_linewidth(3)
# ax.spines['left'].set_linewidth(3)
# ax.spines['right'].set_linewidth(3)
# plt.ylim((0., 1.))
# plt.xticks(np.linspace(-2., 4., 4), size=32)
# plt.yticks(np.linspace(0., 1., 6), size=32)
#
# ax.set_xlabel('reward', size=40)
# ax.set_ylabel('CDF', size=40)
#
# plt.grid()
# plt.legend()
# fig.savefig('figures/reward_cdf.pdf', bbox_inches='tight', format='pdf')

# methods = ['LR', 'LR+VMP', 'NG', 'NG+VMP']
# colors = ['lightcoral', 'skyblue', 'orange', 'cyan']
# methods2filename = {'LR':'lr', 'LR+VMP':'lrvmp',
#                     'NG':'ng', 'NG+VMP':'ngvmp'}
# file_template = 'saved_results/{0}_reward_{1}_{2}_hsdpa'
# file_template = 'saved_results/{0}_reward_{1}_{2}_fcc'
# file_template = 'saved_results/{0}_reward_{1}_{2}_oboe'
# tile_confs = [(2, 4), (4, 6), (6, 8)]
# tile_confs = [(4, 6)]
# psnr_result = np.zeros((len(methods), len(tile_confs)))
# rebuf_result = np.zeros((len(methods), len(tile_confs)))
# 
# fig = plt.figure(figsize=(10, 7))
# ax = fig.add_subplot(111)
#
# for method_id in range(len(methods)):
#     method = methods[method_id]
#     x, y, y_err = [], [], []
#     for tile_conf_id in range(len(tile_confs)):
#         (h, w) = tile_confs[tile_conf_id]
#         psnrs = []
#         rebufs = []
#         with open(file_template.format(methods2filename[method], h, w) , 'r') as f:
#             for line in f:
#                 parser = line.split()
#                 psnr = float(parser[0])
#                 rebuf = float(parser[1])
#                 psnrs.append(psnr)
#                 rebufs.append(rebuf / 2000 * 100)
#         psnrs = np.array(psnrs)
#         rebufs = np.array(rebufs)
#         (psnr_low, psnr_high) = st.t.interval(alpha=0.95, df=len(psnrs) - 1,
#                                               loc=np.mean(psnrs), scale=st.sem(psnrs))
#         (rebuf_low, rebuf_high) = st.t.interval(alpha=0.95, df=len(rebufs) - 1,
#                                               loc=np.mean(rebufs), scale=st.sem(rebufs))
#         psnr_mean = (psnr_low + psnr_high) / 2
#         rebuf_mean = (rebuf_low + rebuf_high) / 2
#         x.append(tile_conf_id + 1 + (method_id - 1.5) * 0.2)
#         print(psnr_mean, rebuf_mean)
#         y.append(psnr_mean)
#         y_err.append(psnr_mean - psnr_low)
        # y.append(rebuf_mean)
        # y_err.append(rebuf_mean - rebuf_low)
#     plt.bar(x, y, color=colors[method_id], width=0.2, label=method)
#     plt.errorbar(x, y, y_err, fmt='o', linewidth=2, capsize=6, color='black')
#
# ax.spines['bottom'].set_linewidth(3)
# ax.spines['top'].set_linewidth(3)
# ax.spines['left'].set_linewidth(3)
# ax.spines['right'].set_linewidth(3)
#
# plt.xticks(np.linspace(1, 3, 3), ['2x4', '4x6', '6x8'], size=32)
# plt.ylim((25., 41.))
# plt.yticks(np.linspace(25., 40., 6), size=32)
# plt.ylim((30., 45.))
# plt.yticks(np.linspace(30., 45., 6), size=32)
# # plt.ylim((0., 20.))
# # plt.yticks(np.linspace(0., 20., 6), size=32)
# # plt.ylim((0., 6.5))
# # plt.yticks(np.linspace(0., 6., 4), size=32)
#
# ax.set_ylabel('Video Quality (dB)', size=40)
# # ax.set_ylabel('Rebuf Ratio (%)', size=40)
# ax.set_xlabel('Tile Configuration', size=40)
# plt.grid()
# plt.legend()
# fig.savefig('figures/psnr_oboe.pdf', bbox_inches='tight', format='pdf')
# fig.savefig('figures/rebuf_oboe.pdf', bbox_inches='tight', format='pdf')
# fig.savefig('figures/psnr_hsdpa.pdf', bbox_inches='tight', format='pdf')
# fig.savefig('figures/rebuf_hsdpa.pdf', bbox_inches='tight', format='pdf')