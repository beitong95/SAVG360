import json
import csv
import os
import glob
import cv2
import math
import numpy as np
import Equirec2Perspec as E2P

from parser import args
from utils import *


video_ids = [5, 3, 15, 12, 5, 8, 1, 7, #ragini
             10, 11, 4, 12, 5, 9, 1, 15, #cody
             14, 15, 3, 2, 8, 1, 11, 10, #beitong
             14, 1, 12, 8, 10, 13, 15, 11, #bo
             10, 12, 2, 6, 4, 5, 13, 8, #zhe
             14, 3, 7, 12, 5, 13, 4, 8, #xiaoyang
             5, 11, 10, 8, 12, 6, 3, 9, #jude
             14, 4, 2, 9, 7, 13, 15, 1, #lucas
             12, 15, 6, 8, 1, 3, 5, 7, #klara
             10, 8, 12, 2, 7, 1, 15, 5, #yinjie
             9, 1, 13, 5, 14, 4, 7, 11 #kai
             ]
user_ids = ['Ragini', 'cody', 'beitong', 'bo', 'zhe', 'xiaoyang',
            'jude', 'Lucas', 'klara', 'yinjie', 'kai']

def get_lonlat(x0, y0, x1, y1):
    if math.isnan(x1) and math.isnan(y1):
        return (float('nan'), float('nan'))
    pixel_x = int(((y1 - args.min_y) / (args.max_y - args.min_y)) * args.sal_h)
    pixel_y = int(((x1 - args.min_x) / (args.max_x - args.min_x)) * args.sal_w)
    if pixel_x < 0 or pixel_x >= args.sal_h:
        return (float('nan'), float('nan'))
    if pixel_y < 0 or pixel_y >= args.sal_w:
        return (float('nan'), float('nan'))
    lonlat = E2P.center2fov(args.fov_span, x0, y0, args.sal_h, args.sal_w)
    fov_pos = E2P.lonlat2XY(lonlat, shape=(args.sal_h, args.sal_w))
    (theta, phi) = xy2lonlat(fov_pos[pixel_x, pixel_y, 1], fov_pos[pixel_x, pixel_y, 0],
                             args.sal_h, args.sal_w)
    # print((x0, y0), (theta, phi))
    return (theta, phi)


def get_eye_data(user_id, filename_id, x, y):
    cur_frameid = 0
    eye_x, eye_y = [], []
    csv_filename = 'output/{0}/{1}.csv'.format(user_id, filename_id)
    fps = 30.0
    if args.video_played in [4, 8, 10, 13]:
        fps = 25.0
    elif args.video_played in [11, 15]:
        fps = 15.0
    with open(csv_filename, 'r') as f_csv:
        data = csv.reader(f_csv)
        timestamp = 0.0
        for row in data:
            if row[0] == 'timestamp':
                continue
            timestamp += 1 / 60.0
            right_x, right_y = float(row[1]), float(row[2])
            left_x, left_y = float(row[3]), float(row[4])
            if timestamp >= (cur_frameid + 1) * (1.0 / fps):
                if cur_frameid >= len(x):
                    continue
                theta, phi = x[cur_frameid], y[cur_frameid]
                if theta == 999:
                    theta, phi = loc_interpolate(x, y, cur_frameid, args)
                if args.data_set == 'user_study':
                    phi = -phi
                lonlat_right = get_lonlat(theta, phi, right_x, right_y)
                lonlat_left = get_lonlat(theta, phi, left_x, left_y)
                if math.isnan(lonlat_right[0]) and math.isnan(lonlat_left[0]):
                    eye_x.append(theta)
                    eye_y.append(phi)
                elif math.isnan(lonlat_right[0]):
                    eye_x.append(lonlat_left[0])
                    eye_y.append(lonlat_left[1])
                elif math.isnan(lonlat_left[0]):
                    eye_x.append(lonlat_right[0])
                    eye_y.append(lonlat_right[1])
                else:
                    eye_x.append((lonlat_left[0] + lonlat_right[0]) * 0.5)
                    eye_y.append((lonlat_left[1] + lonlat_right[1]) * 0.5)
                cur_frameid = cur_frameid + 1
                if cur_frameid % 1000 == 0:
                    template = 'Finishes loading eye tracking data {0}{1} for {2} frames'
                    print(template.format(user_id, filename_id, cur_frameid))
        return eye_x, eye_y


def readin():
    with open(args.user_trace_file, 'r') as f:
        data = json.load(f)
        for x in data:
            if x['type'] == "table":
                trajectory_data = x['data']
    global_view_trajs = {}
    # n_frames = []
    # for target_id in range(1, 16):
    #     for i in range(len(trajectory_data)):
    #         data_point = trajectory_data[i]
    #         x = data_point['x'].lstrip('[').rstrip(']').split(',')
    #         y = data_point['y'].lstrip('[').rstrip(']').split(',')
    #         x = [float(str) for str in x]
    #         y = [float(str) for str in y]
    #         traj_id = i // 2
    #         video_id = video_ids[traj_id]
    #         if video_id == target_id:
    #             print(target_id, len(x))
    #             n_frames.append(len(x))
    #             break
    # print(sum(n_frames), sum(sorted(n_frames)[4:]))
    for i in range(len(trajectory_data)):
        data_point = trajectory_data[i]
        x = data_point['x'].lstrip('[').rstrip(']').split(',')
        y = data_point['y'].lstrip('[').rstrip(']').split(',')
        x = [float(str) for str in x]
        y = [float(str) for str in y]
        traj_id = i // 2
        video_id = video_ids[traj_id]
        if video_id != args.video_played:
            continue
        user_id = user_ids[traj_id // 8]
        filename_id = traj_id % 8 + 1
        # videoName = 'v' + str(videoId + 1)
        if data_point['step'] == "2":
            if video_id not in global_view_trajs.keys():
                global_view_trajs[video_id] = []
            eye_x, eye_y = get_eye_data(user_id, filename_id, x, y)
            if len(eye_x) >= 100:
                global_view_trajs[video_id].append((eye_x, eye_y))
    return global_view_trajs


def saliency_build(trajs, video_name, input_path=args.public_frame_path,
                   output_path=args.sal_gt_path):
    frame_dir = input_path + os.sep + str(video_name)
    frame_list = sorted(glob.glob(frame_dir + os.sep + '*'))
    frame = cv2.imread(frame_list[0], cv2.IMREAD_COLOR)
    h, w, _ = frame.shape
    if not os.path.exists(frame_dir):
        return

    saliency_maps = []
    saliency_max = 0.0
    num_frames = len(trajs[video_name][0][0])
    for i in range(num_frames):
        hdmv_points = []
        for (x, y) in trajs[video_name]:
            theta, phi = x[i], y[i]
            # if theta == 999:
            #     theta, phi = loc_interpolate(x, y, i, args)
            # if args.data_set == 'user_study':
            #     phi = -phi
            hdmv_points.append((theta, phi))
        smap = np.zeros((args.sal_h, args.sal_w))
        if i % args.key_frame_interval == 0:
            for x in range(args.sal_h):
                for y in range(args.sal_w):
                    lonlatxy = xy2lonlat(x, y, args.sal_h, args.sal_w)
                    for lonlat0 in hdmv_points:
                        deg_distance = get_deg_distance(lonlatxy, lonlat0)
                        smap[x][y] += gaussian_from_distance(deg_distance)
            saliency_max = max(saliency_max, np.max(smap))
        else:
            smap = saliency_maps[-1]
        saliency_maps.append(smap)
        if i % 100 == 0:
            template = 'video {0} finishes building saliency map for {1}/{2} frames'
            print(template.format(video_name, i, num_frames))
    for i in range(num_frames):
        saliency_maps[i] = saliency_maps[i] / saliency_max
    fps = 30.0
    if args.video_played in [4, 8, 10, 13]:
        fps = 25.0
    elif args.video_played in [11, 15]:
        fps = 15.0
    demo_filename_template = 'saliency_demo_{0}.mp4'
    # out = cv2.VideoWriter(demo_filename_template.format(video_name),
    #                       cv2.VideoWriter_fourcc(*'DIVX'), fps, (w, h))
    sal_gt_dir = output_path + os.sep + str(video_name)
    if not os.path.exists(sal_gt_dir):
        os.mkdir(sal_gt_dir)
    for i in range(num_frames):
        filename = frame_list[i].split('/')[-1].split('.')[0]
        # frame = cv2.imread(frame_list[i], cv2.IMREAD_COLOR)
        saliency_maps[i] = saliency_maps[i] / saliency_max
        smap = (saliency_maps[i] * 255).astype("uint8")
        smap = cv2.resize(smap, (w, h), interpolation=cv2.INTER_CUBIC)
        np.save(sal_gt_dir + os.sep + filename, saliency_maps[i])
        # output = (frame * 0.3).astype("uint8")
        # output[..., 2] = (output[..., 2] + 0.7 * smap).astype("uint8")
        # out.write(output)
    return saliency_maps


traj = readin()
saliency_build(traj, args.video_played, args.frame_path)

