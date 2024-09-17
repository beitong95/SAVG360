import glob
import random
import os
import numpy as np
import torch
import torch.nn.functional as F
import json
from model import CNN, LR
from combined_vp import load_vp_data, Watching_Mode_Predictor, \
    Viewport_Predictor_LR, Viewport_Predictor_NG
from env import Environment
from parser import args
from Equirec2Perspec import get_tile_in_FoV
from utils import *

random.seed(1)
torch.manual_seed(1)


def vp_prediction(video_id, vp_trace, guidance_data,
                  tile_h=args.tile_h, tile_w=args.tile_w, vp_method=args.vp_method):
    metric1, metric2 = [], []
    num_chunks = len(guidance_data["global_guidance"])
    vp_model = None
    if vp_method == 'LR':
        vp_model = Viewport_Predictor_LR(guidance_data, tile_h, tile_w, args.threshold)
    elif vp_method == 'NG':
        vp_model = Viewport_Predictor_NG(video_id, guidance_data, tile_h, tile_w, args.threshold)
    with torch.no_grad():
        for (i, input_follow, _, _, _, input_vp, label_vp) in vp_trace:
            # tile_follow = vp_model.get_tile_follow(i, input_vp, args.skips)
            if i + args.skips >= num_chunks:
                break
            # start_vp = input_vp[0]
            # end_vp = label_vp[args.skips * 5 - 1]
            # dist = lonlat2dist(start_vp[0].item(), start_vp[1].item(),
            #                end_vp[0].item(), end_vp[1].item())
            # # print(dist)
            # if  dist< 0.0001:
            #     continue
            tile_vp = vp_model.get_tile_vp(input_vp, args.skips)
            if i == 0:
                tile_prob = vp_model.predict(i, input_follow, input_vp, args.skips, prob_follow=0.5)
            else:
                tile_prob = vp_model.predict(i, input_follow, input_vp, args.skips)

            tile_label = np.ones((tile_h, tile_w))
            for _ in label_vp[args.skips * 5 - 1: (args.skips + 1) * 5]:
                vp = (_[0].item() * 360, _[1].item() * 180)
                tile_label = tile_label * (1 - get_tile_in_FoV((vp)))
            tile_label = 1 - tile_label

            # acc_follow = np.sum(tile_follow * tile_label) / np.sum(tile_label)
            # rec_follow = np.sum(tile_follow * tile_label) / np.sum(tile_follow)
            acc_vp = np.sum(tile_vp * tile_label) / np.sum(tile_label)
            rec_vp = np.sum(tile_vp * tile_label) / np.sum(tile_vp)
            # print(prob_follow, acc_follow, acc_vp)

            acc = np.sum(tile_prob * tile_label) / np.sum(tile_label)
            rec = np.sum(tile_prob * tile_label) / np.sum(tile_prob)
            metric1.append(acc)
            metric2.append(acc_vp)

            # if label_follow[0].item() == 1:
            # print(i, prediction_follow, prediction_vp, label_follow, label_vp)
    acc_1, acc_2 = 1.0, 1.0
    if len(metric1) > 0:
        acc_1 = sum(metric1) / len(metric1)
    if len(metric2) > 0:
        acc_2 = sum(metric2) / len(metric2)
    return acc_1, acc_2
    # return sum(metric1) / len(metric1), sum(metric2) / len(metric2)


def mode_prediction(video_id, vp_trace, guidance_data, TP, FP, TN, FN,
                  tile_h=args.tile_h, tile_w=args.tile_w):
    # metric1, metric2 = [], []
    num_chunks = len(guidance_data["global_guidance"])
    model = Watching_Mode_Predictor(guidance_data, tile_h, tile_w)
    with torch.no_grad():
        for (i, input_follow, label_follow_1, label_follow_2, label_follow_3, _, _) in vp_trace:
            label_follow = [label_follow_1.item(), label_follow_2.item(), label_follow_3.item()]
            if i + args.skips >= num_chunks:
                break
            if i == 0:
                continue
            prob_follow = model.get_prob(input_follow, args.skips)
            if prob_follow > 0.5:
                if label_follow[args.skips - 1] == 1.0:
                    TP += 1
                else:
                    FP += 1
            else:
                if label_follow[args.skips - 1] == 1.0:
                    FN += 1
                else:
                    TN += 1
    return TP, FP, TN, FN


def load_bw_traces(bandwidth_trace_files=args.bandwidth_trace_files):
    files = sorted(glob.glob(bandwidth_trace_files + os.sep + '*'))
    bw_traces = []
    avg_bw = []
    for file in files:
        bw, time = [], []
        with open(file, 'r') as f:
            for line in f:
                parse = line.split()
                bw.append(float(parse[1]))
                time.append(float(parse[0]))
        bw_traces.append((bw, time))
        # print(sum(bw) / len(bw))
        avg_bw.append(sum(bw)/len(bw))
    print(sum(avg_bw) / len(avg_bw))
    return bw_traces


def load_video_traces():
    video_trace_file = '/home/bizon/mi360dataset/final_{0}_{1}.txt'.\
        format(args.tile_h, args.tile_w)
    video_sizes, qualities, max_chunk = {}, {}, {}
    with open(video_trace_file, "r") as f:
        no_lines = 0
        for line in f:
            parser = line.split()
            filename_parser = parser[0].split('/')
            video_parser = filename_parser[4].split('_')
            tile_chunk_parser = filename_parser[6].split('.')[0].split('_')
            video_id, quality_level = int(video_parser[0]), video_parser[1]
            tile_id = int(tile_chunk_parser[2]) * args.tile_w + int(tile_chunk_parser[1])
            chunk_id = int(int(tile_chunk_parser[-1]) / (args.video_chunk_len / args.ms_in_s))
            # print(video_id, quality_level, tile_id, chunk_id)
            if video_id not in video_sizes:
                video_sizes[video_id] = {}
                qualities[video_id] = {}
                max_chunk[video_id] = 0
            video_sizes[video_id][(chunk_id, quality_level, tile_id)] = int(parser[1]) // args.bits_in_byte
            qualities[video_id][(chunk_id, quality_level, tile_id)] = min(float(parser[2]), 55.0)
            max_chunk[video_id] = max(max_chunk[video_id], chunk_id)
    return video_sizes, qualities, max_chunk


def get_average(video_sizes, qualities, max_chunk, video_id, quality_level, tile_id):
    sum_video_size, sum_qualities, num_data = 0, 0, 0
    for chunk_id in range(max_chunk):
        if (chunk_id, quality_level, tile_id) in video_sizes[video_id]:
            sum_video_size += video_sizes[video_id][(chunk_id, quality_level, tile_id)]
            sum_qualities += qualities[video_id][(chunk_id, quality_level, tile_id)]
            num_data += 1
    return int(sum_video_size / num_data), sum_qualities / num_data


def tile_bitrate_selection(tile_prob, video_size, video_quality, capacity, tile_h=args.tile_h,
                           tile_w=args.tile_w, quality_levels=args.quality_levels, granularity=10000):
    capacity = round(capacity / granularity)
    num_tiles = tile_h * tile_w
    dp = np.zeros((num_tiles + 1, capacity + 1))
    prev = np.zeros((num_tiles + 1, capacity + 1))
    for i in range(num_tiles):
        for j in range(len(quality_levels)):
            tile_size = round(video_size[i, j] / granularity)
            if tile_size > capacity:
                continue
            prev[i + 1, tile_size:] = np.where(dp[i + 1, tile_size:] >= tile_prob[i] * video_quality[i, j] +
                                               dp[i, : capacity + 1 - tile_size], prev[i + 1, tile_size: ], j)
            dp[i + 1, tile_size: ] = np.maximum(dp[i + 1, tile_size: ], tile_prob[i] * video_quality[i, j] +
                                            dp[i, : capacity + 1 - tile_size])
    # print(dp)
    # print(prev)
    cap = capacity
    selected_levels = []
    max_prob = np.max(tile_prob)
    for i in range(num_tiles, 0, -1):
        # print(i, tile_prob[i - 1], int(prev[i, cap]), dp[i, cap])
        selected_level = int(prev[i, cap])
        # if tile_prob[i - 1] == max_prob and selected_level == 0:
        #     selected_level = 1
        selected_levels = [selected_level] + selected_levels
        cap = cap - round(video_size[i - 1, int(prev[i, cap])]/ granularity)
    return selected_levels


def compute_reward(video_chunk_info, selected_levels, rebuf, throughput, vp_trace,
                   tile_h=args.tile_h, tile_w=args.tile_w):
    tile_label = np.ones((tile_h, tile_w))
    for _ in vp_trace:
        vp = (_[0].item() * 360, _[1].item() * 180)
        tile_label = tile_label * (1 - get_tile_in_FoV((vp)))
    tile_label = 1 - tile_label
    tile_label = np.reshape(tile_label, tile_h * tile_w)

    total_rebuf, total_quality, tiles_viewed = rebuf, 0, 0
    (video_size, video_quality) = video_chunk_info
    for tile_no in range(tile_h * tile_w):
        if tile_label[tile_no] == 1:
            tiles_viewed += 1
            selected_level = selected_levels[tile_no]
            total_quality += video_quality[tile_no, selected_level]
            # if selected_level == 0:
            #     total_quality += video_quality[tile_no, 1]
            #     total_rebuf += video_size[tile_no, 1] / throughput / args.m_in_k / args.ms_in_s
            # else:
            #     total_quality += video_quality[tile_no, selected_level]
    return total_quality / tiles_viewed, total_rebuf * args.ms_in_s


def streaming(video_id, bw, time, ml_trace, vp_data, guidance_data,
              video_sizes, qualities, tile_h=args.tile_h, tile_w=args.tile_w,
              default_quality=1, vp_method=args.vp_method):
    num_chunks = len(vp_data)

    net_env = Environment(time=time, bw=bw, num_chunks=num_chunks,
                          video_sizes=video_sizes[video_id], qualities=qualities[video_id])
    if vp_method == 'LR':
        vp_model = Viewport_Predictor_LR(guidance_data, tile_h, tile_w, args.threshold)
    elif vp_method == 'NG':
        vp_model = Viewport_Predictor_NG(video_id, guidance_data, tile_h, tile_w, args.threshold)
    player_counter = 0

    time_stamp = 0
    selected_levels = []
    for _ in range(tile_h * tile_w):
        selected_levels.append(default_quality)

    accumulated_psnr, accumulated_rebuf = 0, 0
    past_throughputs, past_throughput_predictions, past_errors = [], [], []

    while True:
        delay, sleep_time, buffer_size, rebuf, video_chunk_size, this_chunk_info, \
        next_chunk_info, end_of_video, video_chunk_remain = net_env.get_video_chunk(selected_levels)
        time_stamp += delay # in ms
        time_stamp += sleep_time # in ms

        throughput = float(video_chunk_size) / float(delay) / args.m_in_k
        past_throughputs.append(throughput)
        curr_error = 0
        if len(past_throughput_predictions) > 0:
            curr_error = abs(past_throughput_predictions[-1] - throughput) / float(throughput)
        past_errors.append(curr_error)

        psnr, rebuf = compute_reward(this_chunk_info, selected_levels, rebuf, throughput, vp_data[player_counter])

        accumulated_psnr += psnr
        if player_counter > 0:
            accumulated_rebuf += rebuf

        if end_of_video:
            break
        player_counter += 1

        chunk_skip = 1
        if buffer_size > 2.0:
            chunk_skip += 1
        (j, input_follow, _, _, _, input_vp, _) = ml_trace[player_counter - chunk_skip]
        tile_vp = vp_model.get_tile_vp(input_vp, chunk_skip)
        tile_vp = np.reshape(tile_vp, tile_h * tile_w)
        if j == 0:
            tile_prob = vp_model.predict(0, input_follow, input_vp, args.skips, prob_follow=0.5)
        else:
            tile_prob = vp_model.predict(j, input_follow, input_vp, args.skips)
        tile_prob = np.reshape(tile_prob, tile_h * tile_w)
        (video_size, video_quality) = next_chunk_info

        max_error = float(max(past_errors[-5: ]))
        future_bandwidth = harmonic_avg(past_throughputs[-5: ]) / (1 + max_error)  # robustMPC here
        past_throughput_predictions.append(future_bandwidth)

        capacity = int(future_bandwidth * buffer_size * args.b_in_mb)
        # capacity = int(throughput * buffer_size * args.b_in_mb)
        # selected_levels = tile_bitrate_selection(tile_prob, video_size, video_quality, capacity)
        selected_levels = tile_bitrate_selection(tile_vp, video_size, video_quality, capacity)
        # print(capacity, tile_prob, video_size[1], selected_levels)

    return accumulated_psnr / num_chunks, accumulated_rebuf / num_chunks


if __name__ == '__main__':
    test_data, guidance_data = load_vp_data()

    metric1, metric2 = [], []
    TP, FP, TN, FN = 0, 0, 0, 0
    # for vp_trace in test_data:
        # TP, FP, TN, FN = mode_prediction(vp_trace["video_id"], vp_trace["ml_data"],
        #                                  guidance_data[vp_trace["video_id"]], TP, FP, TN, FN)
        # m1, m2 = vp_prediction(vp_trace["video_id"], vp_trace["ml_data"],
        #                        guidance_data[vp_trace["video_id"]])
        # metric1.append(m1)
        # metric2.append(m2)
    # total = TP + FP + TN + FN
    # print(TP / total, FP / total, TN / total, FN / total)
    # print(sum(metric1) / len(metric1), sum(metric2) / len(metric2))
    # with open('saved_results/prediction_acc_lrwmp_' + str(args.skips * 2) , 'w') as f:
    #     for m1 in metric1:
    #         acc = round(m1, 4)
    #         print(acc, file=f)
    # with open('saved_results/prediction_acc_lr_' + str(args.skips * 2) , 'w') as f:
    #     for m2 in metric2:
    #         acc = round(m2, 4)
    #         print(acc, file=f)

    bw_traces = load_bw_traces()
    video_sizes, qualities, max_chunk = load_video_traces()
    for vp_trace in test_data:
        num_chunks = len(vp_trace["vp_data"])
        video_id = vp_trace["video_id"]
        for chunk_id in range(num_chunks):
            for level_id in range(1, len(args.quality_levels)):
                for tile_id in range(args.tile_w * args.tile_h):
                    quality_level = args.quality_levels[level_id]
                    if (chunk_id, quality_level, tile_id) not in video_sizes[video_id]:
                        avg_size, avg_quality = get_average(video_sizes, qualities, max_chunk[video_id],
                                    video_id, quality_level, tile_id)
                        video_sizes[video_id][(chunk_id, quality_level, tile_id)] = avg_size
                        qualities[video_id][(chunk_id, quality_level, tile_id)] = avg_quality
    rewards = []
    num_vp_traces = len(test_data)
    for i in range(len(bw_traces)):
        (bw, time) = bw_traces[i]
        vp_trace_id = random.randint(0, num_vp_traces - 1)
        vp_trace = test_data[vp_trace_id]
        psnr, rebuf = streaming(vp_trace["video_id"], bw, time, vp_trace["ml_data"],
                           vp_trace["vp_data"], guidance_data[vp_trace["video_id"]],
                           video_sizes, qualities)
        rewards.append((psnr, rebuf))
        template = "bw trace ({0}/{1}), vp trace {2}, psnr: {3}, rebuf: {4}"
        print(template.format(i + 1, len(bw_traces), vp_trace_id, round(psnr, 2), round(rebuf, 2)))
    # print("final reward:", sum(rewards) / len(rewards))
    result_file = "saved_results/ng_reward_{0}_{1}_oboe".format(args.tile_h, args.tile_w)
    with open(result_file, 'w') as f:
        for reward in rewards:
            (psnr, rebuf) = reward
            rounded_psnr = round(psnr, 2)
            rounded_rebuf = round(rebuf, 2)
            print(str(rounded_psnr) + ' ' + str(rounded_rebuf), file=f)
