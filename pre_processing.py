import os
import glob
import time
import cv2
import csv
import numpy as np
from utils import *
from parser import args
import Equirec2Perspec as E2P


def saliency_build(trajs, video_name, input_path=args.public_frame_path):
    frame_dir = input_path + os.sep + str(video_name + 1)
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
            if theta == 999:
                theta, phi = loc_interpolate(x, y, i, args)
            if args.data_set == 'user_study':
                phi = -phi
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
    # demo_filename_template = 'demo_{0}.mp4'
    # out = cv2.VideoWriter(demo_filename_template.format(video_name),
    #                       cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))
    # for i in range(num_frames):
    #     frame = cv2.imread(frame_list[i], cv2.IMREAD_COLOR)
        # saliency_maps[i] = saliency_maps[i] / saliency_max
        # smap = (saliency_maps[i] * 255).astype("uint8")
        # smap = cv2.resize(smap, (w, h), interpolation=cv2.INTER_CUBIC)
        # output = (frame * 0.3).astype("uint8")
        # output[..., 2] = (output[..., 2] + 0.7 * smap).astype("uint8")
        # out.write(output)
    return saliency_maps


def saliency_readin(frame_path, saliency_path, video_name,
                    key_frame_interval=args.key_frame_interval):
    frame_dir = frame_path + os.sep + str(video_name)
    frame_list = sorted(glob.glob(frame_dir + os.sep + '*'))
    saliency_dir = saliency_path + os.sep + str(video_name)
    saliency_list = sorted(glob.glob(saliency_dir + os.sep + '*'))[:-1]

    saliency_maps = []
    saliency_max = 0.0
    num_frames = min(len(frame_list), int(args.video_length * args.fps))
    len_sal_list = len(saliency_list)

    for i in range(num_frames):
        sal_id = i - 4
        if sal_id < 0:
            sal_id = 0
        elif sal_id >= len_sal_list:
            sal_id = len_sal_list - 1
        orig_sal = np.load(saliency_list[sal_id])
        sal_id_2 = sal_id - int(args.fps)
        if sal_id_2 < 0:
            sal_id_2 = sal_id + int(args.fps)
        ref_sal = np.load(saliency_list[sal_id_2])
        zero_mask = np.zeros_like(orig_sal)
        concentrated_sal = np.array((orig_sal - ref_sal, zero_mask))
        del_sal = concentrated_sal.max(axis=0)
        # smap = cv2.resize(del_sal, (args.sal_w, args.sal_h), interpolation=cv2.INTER_CUBIC)
        smap = cv2.resize(orig_sal, (args.sal_w, args.sal_h), interpolation=cv2.INTER_CUBIC)
        saliency_maps.append(smap)
        saliency_max = max(saliency_max, np.max(smap))
        if i % 100 == 0:
            template = 'video {0} finishes reading saliency map for {1}/{2} frames'
            print(template.format(video_name, i, num_frames))

    key_saliency_maps = []
    accumulated_frames = 0
    accumulated_saliency = np.zeros_like(saliency_maps[0])
    for i in range(num_frames):
        saliency_maps[i] = saliency_maps[i] / saliency_max
        accumulated_saliency = accumulated_saliency + saliency_maps[i]
        accumulated_frames = accumulated_frames + 1
        if (i + 1) % key_frame_interval == 0:
            key_saliency_maps.append(accumulated_saliency / accumulated_frames)
            accumulated_frames = 0
    if accumulated_frames != 0:
        key_saliency_maps.append(accumulated_saliency / accumulated_frames)
    for i in range(len(key_saliency_maps)):
        saliency_maps[i * key_frame_interval] = key_saliency_maps[i]
    return saliency_maps


def scanpath_generating(saliency_maps, pixel2fov,
                        key_frame_interval=args.key_frame_interval):
    # dynamic_programming
    num_frames = len(saliency_maps)
    h, w = args.sal_h, args.sal_w
    num_key_frames = (num_frames - 1) // key_frame_interval + 1
    last_key_frame = (num_key_frames - 1) * key_frame_interval
    assert (args.interval * args.fps) % key_frame_interval <= 1e-3 or \
           (args.interval * args.fps) % key_frame_interval >= key_frame_interval - 1e-3
    seg_frame_interval = round((args.interval * args.fps) / key_frame_interval)
    score = np.zeros((num_key_frames, h, w))
    seg_score_storyline = np.zeros((num_key_frames, h, w))
    seg_score_best = np.zeros((num_key_frames, h, w))
    next_x = np.zeros((num_key_frames, h, w)).astype('int')
    next_y = np.zeros((num_key_frames, h, w)).astype('int')
    seg_next_x = np.zeros((num_key_frames, h, w)).astype('int')
    seg_next_y = np.zeros((num_key_frames, h, w)).astype('int')
    frame_sal = np.zeros((2, h, w, h, w))
    for x in range(h):
        for y in range(w):
            fov_pos_xy, pixel_weight_xy = pixel2fov.fov_pos[(x, y)], pixel2fov.pixel_weight[(x, y)]
            orig_sal_map = cv2.remap(saliency_maps[last_key_frame], fov_pos_xy[..., 0],
                          fov_pos_xy[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
            frame_sal[(num_key_frames - 1) % 2][x][y] = orig_sal_map
            score[num_key_frames - 1][x][y] = np.sum(orig_sal_map * pixel_weight_xy) / np.sum(pixel_weight_xy)
            seg_score_storyline[num_key_frames - 1][x][y] = score[num_key_frames - 1][x][y]
            seg_score_best[num_key_frames - 1][x][y] = score[num_key_frames - 1][x][y]
            next_x[num_key_frames - 1][x][y], next_y[num_key_frames - 1][x][y] = x, y
            seg_next_x[num_key_frames - 1][x][y], seg_next_y[num_key_frames - 1][x][y] = x, y
    tic = time.time()
    for i in range(num_key_frames - 2, -1, -1):
        for x0 in range(h):
            for y0 in range(w):
                fov_pos_xy0, pixel_weight_xy0 = pixel2fov.fov_pos[(x0, y0)], pixel2fov.pixel_weight[(x0, y0)]
                orig_sal_map = cv2.remap(saliency_maps[i * key_frame_interval], fov_pos_xy0[..., 0],
                                                     fov_pos_xy0[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
                frame_sal[i % 2][x0][y0] = orig_sal_map
                sal_average = np.sum(orig_sal_map * pixel_weight_xy0) / np.sum(pixel_weight_xy0)
                if (i + 1) % seg_frame_interval == 0:
                    seg_score_storyline[i][x0][y0] = sal_average
                    seg_score_best[i][x0][y0] = sal_average
                    seg_next_x[i][x0][y0], seg_next_y[i][x0][y0] = x0, y0
                for x1 in range(h):
                    for y1 in range(w):
                        sal_vibration, vp_movement = 0.0, pixel_dist(x0, y0, x1, y1)
                        if vp_movement >= 0.5:
                            continue
                        if score[i + 1][x1][y1] * args.gamma + sal_average - \
                                args.alpha1 * vp_movement > score[i][x0][y0]:
                            # sal_vibration = np.sqrt(np.average((frame_sal[i % 2][x0][y0] -
                            #                             frame_sal[(i + 1) % 2][x1][y1]) ** 2))
                            temp_score = score[i + 1][x1][y1] * args.gamma + sal_average - \
                                         args.alpha0 * sal_vibration - args.alpha1 * vp_movement
                            if temp_score > score[i][x0][y0]:
                                # print(sal_average, sal_vibration, vp_movement)
                                score[i][x0][y0] = temp_score
                                next_x[i][x0][y0], next_y[i][x0][y0] = x1, y1
                                if (i + 1) % seg_frame_interval != 0:
                                    seg_score_storyline[i][x0][y0] = \
                                        seg_score_storyline[i + 1][x1][y1] + sal_average - \
                                        args.alpha0 * sal_vibration - args.alpha2 * vp_movement
                        if seg_score_best[i + 1][x1][y1] + sal_average - args.alpha2 * \
                                vp_movement > seg_score_best[i][x0][y0]:
                            # if sal_vibration == -1:
                            #     sal_vibration = np.sqrt(np.average((frame_sal[i % 2][x0][y0] -
                            #                                         frame_sal[(i + 1) % 2][x1][y1]) ** 2))
                            if (i + 1) % seg_frame_interval != 0:
                                temp_seg_score = seg_score_best[i + 1][x1][y1] + sal_average - \
                                                 args.alpha0 * sal_vibration - args.alpha2 * vp_movement
                                if temp_seg_score > seg_score_best[i][x0][y0]:
                                    seg_score_best[i][x0][y0] = temp_seg_score
                                    seg_next_x[i][x0][y0] = x1
                                    seg_next_y[i][x0][y0] = y1
        if i % 1 == 0:
            template = 'generating scanpath remains {0}/{1} keyframes, time used: {2} minites'
            print(template.format(i, num_key_frames, (time.time() - tic) / 60))

    lookup_template = 'lookup_tables' + os.sep + '{0}_{1}.csv'
    print('writing global guide table')
    headers = ['key_frame_idx', 'x', 'y', 'score', 'next_x', 'next_y',
               'seg_next_x', 'seg_next_y']
    rows = []
    for i in range(num_key_frames):
        for x in range(h):
            for y in range(w):
                rows.append([i, x, y, score[i][x][y], next_x[i][x][y],
                             next_y[i][x][y], seg_next_x[i][x][y], seg_next_y[i][x][y]])
    with open(lookup_template.format('global', args.video_played), 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)

    print('writing segment guide table')
    headers = ['seg_idx', 'x', 'y', 'global_score', 'seg_score']
    rows = []
    for i in range(num_key_frames):
        if i % seg_frame_interval != 0:
            continue
        seg_id = i // seg_frame_interval
        for x in range(h):
            for y in range(w):
                rows.append([int(seg_id), x, y, seg_score_storyline[i][x][y],
                             seg_score_best[i][x][y]])
    with open(lookup_template.format('segment', args.video_played), 'w') as f:
        f_csv = csv.writer(f)
        f_csv.writerow(headers)
        f_csv.writerows(rows)


# global_view_trajectories = readin_trajectories(args)
pixel2fov = E2P.Pixel2FoV(args.sal_h, args.sal_w)
saliency_maps = saliency_readin(args.frame_path, args.sal_path, args.video_played)
# saliency_maps = saliency_build(global_view_trajectories, args.video_played,
#                                args.frame_path)
scanpath_generating(saliency_maps, pixel2fov)
