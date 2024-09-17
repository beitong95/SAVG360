import os
import glob
import time
import cv2
import numpy as np
from utils import *
from parser import args
from copy import deepcopy
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


def scanpath_generating(saliency_maps, pixel2fov,
                        key_frame_interval=args.key_frame_interval):
    # dynamic_programming
    num_frames = len(saliency_maps)
    h, w = args.sal_h, args.sal_w
    num_key_frames = (num_frames - 1) // key_frame_interval + 1
    last_key_frame = (num_key_frames - 1) * key_frame_interval
    assert (args.interval * args.fps) % key_frame_interval == 0
    seg_frame_interval = (args.interval * args.fps) // key_frame_interval
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
            fov_pos_xy = pixel2fov.fov_pos[(x, y)]
            frame_sal[(num_key_frames - 1) % 2][x][y] = \
                cv2.remap(saliency_maps[last_key_frame], fov_pos_xy[..., 0],
                          fov_pos_xy[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
            score[num_key_frames - 1][x][y] = np.average(frame_sal[(num_key_frames - 1) % 2][x][y])
            seg_score_storyline[num_key_frames - 1][x][y] = score[num_key_frames - 1][x][y]
            seg_score_best[num_key_frames - 1][x][y] = score[num_key_frames - 1][x][y]
            seg_next_x[num_key_frames - 1][x][y], seg_next_y[num_key_frames - 1][x][y] = x, y
    tic = time.time()
    for i in range(num_key_frames - 2, 0, -1):
        for x0 in range(h):
            for y0 in range(w):
                fov_pos_xy0 = pixel2fov.fov_pos[(x0, y0)]
                frame_sal[i % 2][x0][y0] = cv2.remap(saliency_maps[i * key_frame_interval], fov_pos_xy0[..., 0],
                                                     fov_pos_xy0[..., 1], cv2.INTER_CUBIC, borderMode=cv2.BORDER_WRAP)
                (theta, phi) = xy2lonlat(x0, y0, args.sal_h, args.sal_w)
                xyz0 = lonlat2xyz(theta, phi)
                sal_average = np.average(frame_sal[i % 2][x0][y0])
                if (i + 1) % seg_frame_interval == 0:
                    seg_score_storyline[i][x0][y0] = sal_average
                    seg_score_best[i][x0][y0] = sal_average
                    seg_next_x[i][x0][y0], seg_next_y[i][x0][y0] = x0, y0
                for x1 in range(h):
                    if abs(x0 - x1) > h // 15 and h - abs(x0 - x1) > h // 15:
                        continue
                    for y1 in range(w):
                        if abs(y0 - y1) > w // 6 and w - abs(y0 - y1) > w // 6:
                            continue
                        (theta, phi) = xy2lonlat(x1, y1, args.sal_h, args.sal_w)
                        xyz1 = lonlat2xyz(theta, phi)
                        vp_movement = np.sqrt(np.sum((xyz0 - xyz1) ** 2))
                        if vp_movement >= 0.2:
                            continue
                        sal_vibration = np.sqrt(np.average((frame_sal[i % 2][x0][y0] -
                                                    frame_sal[(i + 1) % 2][x1][y1]) ** 2))
                        temp_score = score[i + 1][x1][y1] + sal_average - \
                                     args.alpha0 * sal_vibration - args.alpha1 * vp_movement
                        if temp_score > score[i][x0][y0]:
                            score[i][x0][y0] = temp_score
                            next_x[i][x0][y0], next_y[i][x0][y0] = x1, y1
                            if (i + 1) % seg_frame_interval != 0:
                                seg_score_storyline[i][x0][y0] = \
                                    seg_score_storyline[i + 1][x1][y1] + sal_average - \
                                    args.alpha0 * sal_vibration - args.alpha1 * vp_movement
                        if (i + 1) % seg_frame_interval != 0:
                            temp_seg_score = seg_score_best[i + 1][x1][y1] + sal_average - \
                                             args.alpha0 * sal_vibration - args.alpha1 * vp_movement
                            if temp_seg_score > seg_score_best[i][x0][y0]:
                                seg_score_best[i][x0][y0] = temp_seg_score
                                seg_next_x[i][x0][y0] = x1
                                seg_next_y[i][x0][y0] = y1
        if i % 1 == 0:
            template = 'generating scanpath remains {0}/{1} keyframes, time used: {2} minites'
            print(template.format(i, num_key_frames, (time.time() - tic) / 60))

    return score, seg_score_storyline, seg_score_best, next_x, next_y, seg_next_x, seg_next_y, num_frames


def play_back(score, seg_score_storyline, seg_score_best,
              next_x, next_y, seg_next_x, seg_next_y,
              num_frames, input_path=args.frame_path, video_name=args.video_played,
              key_frame_interval=args.key_frame_interval):
    frame_dir = input_path + os.sep + str(video_name + 1)
    frame_list = sorted(glob.glob(frame_dir + os.sep + '*'))
    frame = cv2.imread(frame_list[0], cv2.IMREAD_COLOR)
    h, w = frame.shape[0] // 2, frame.shape[1] // 2

    sal_h, sal_w = args.sal_h, args.sal_w
    num_key_frames = (num_frames - 1) // key_frame_interval + 1
    assert (args.interval * args.fps) % key_frame_interval == 0
    seg_frame_interval = (args.interval * args.fps) // key_frame_interval
    keyframe_trace = []

    max_score_pos = np.argmax(score[0])
    cur_x, cur_y = max_score_pos // sal_w, max_score_pos % sal_w

    seg_idx, key_frame_idx = 0, 0
    while key_frame_idx < num_key_frames:
        seg_start = seg_idx * seg_frame_interval
        seg_end = min(seg_start + seg_frame_interval, num_key_frames)

        seg_windows = []
        for wid in range(1 + args.num_small_windows):
            seg_windows.append([])

        for i in range(seg_start, seg_end):
            seg_windows[0].append((cur_x, cur_y))
            nxt_x = next_x[i][cur_x][cur_y]
            nxt_y = next_y[i][cur_x][cur_y]
            cur_x, cur_y = nxt_x, nxt_y

        seg_scores = []
        for x in range(sal_h):
            for y in range(sal_w):
                seg_scores.append((seg_score_best[seg_start][x][y], x, y))
        seg_scores.sort(reverse=True)

        cur_wid = 1
        occupied_start = [seg_windows[0][0]]
        occupied_end = [seg_windows[0][-1]]
        for (val, x, y) in seg_scores:
            if val < args.beta * seg_score_storyline[seg_start][cur_x][cur_y]:
                break
            if cur_wid > args.num_small_windows:
                break

            occupied_flag = False
            (theta, phi) = xy2lonlat(x, y, args.sal_h, args.sal_w)
            xyz = lonlat2xyz(theta, phi)
            for (x0, y0) in occupied_start:
                (theta, phi) = xy2lonlat(x0, y0, args.sal_h, args.sal_w)
                xyz0 = lonlat2xyz(theta, phi)
                if np.sqrt(np.sum((xyz0 - xyz) ** 2)) <= 1.2:
                    occupied_flag = True
                    break
            if occupied_flag is True:
                continue

            temp_window = []
            temp_x, temp_y = x, y
            for i in range(seg_start, seg_end):
                temp_window.append((temp_x, temp_y))
                nxt_x = seg_next_x[i][temp_x][temp_y]
                nxt_y = seg_next_y[i][temp_x][temp_y]
                temp_x, temp_y = nxt_x, nxt_y

            (end_x, end_y) = temp_window[-1]
            (theta, phi) = xy2lonlat(end_x, end_y, args.sal_h, args.sal_w)
            xyz = lonlat2xyz(theta, phi)
            for (x0, y0) in occupied_end:
                (theta, phi) = xy2lonlat(x0, y0, args.sal_h, args.sal_w)
                xyz0 = lonlat2xyz(theta, phi)
                if np.sqrt(np.sum((xyz0 - xyz) ** 2)) <= 0.2:
                    occupied_flag = True
                    break
            if occupied_flag is False:
                occupied_start.append((x, y))
                occupied_end.append((end_x, end_y))
                seg_windows[cur_wid] = deepcopy(temp_window)
                cur_wid = cur_wid + 1

        for i in range(seg_start, seg_end):
            keyframe_trace.append([])
            for wid in range(1 + args.num_small_windows):
                if seg_windows[wid] != []:
                    keyframe_trace[i].append(seg_windows[wid][i - seg_start])
                else:
                    keyframe_trace[i].append((-1, -1))

        if seg_end == num_key_frames:
            break
        seg_idx = seg_idx + 1
    # for i in range(num_key_frames):
    #     main_window = (cur_x, cur_y)
    #     small_windows = []
    #     seg_scores = []
    #     if i % seg_frame_interval == 0:
    #         for x in range(sal_h):
    #             for y in range(sal_w):
    #                 seg_scores.append((seg_score_best[i][x][y], x, y))
    #         seg_scores.sort(reverse=True)
    #         occupied_start = [(cur_x, cur_y)]
    #         # occupied_end = [(nxt_seg_x, nxt_seg_y)]
    #         for (val, x, y) in seg_scores:
    #             if val < args.beta * seg_score_storyline[i][cur_x][cur_y]:
    #                 break
    #             if len(small_windows) == args.num_small_windows:
    #                 break
    #             occupied_flag = False
    #             (theta, phi) = xy2lonlat(x, y, args.sal_h, args.sal_w)
    #             xyz = lonlat2xyz(theta, phi)
    #             for (x0, y0) in occupied_start:
    #                 (theta, phi) = xy2lonlat(x0, y0, args.sal_h, args.sal_w)
    #                 xyz0 = lonlat2xyz(theta, phi)
    #                 if np.sum((xyz0 - xyz) ** 2) <= 1.0:
    #                     occupied_flag = True
    #                     break
    #             if occupied_flag is False:
    #                 occupied_start.append((x, y))
    #                 small_windows.append((x, y))
    #         while len(small_windows) < args.num_small_windows:
    #             small_windows.append((-1, -1))
    #     else:
    #         for (x, y) in keyframe_trace[i - 1][1:]:
    #             if x == -1:
    #                 small_windows.append((-1, -1))
    #             else:
    #                 small_windows.append((seg_next_x[i - 1][x][y], seg_next_y[i - 1][x][y]))
    #
    #     keyframe_trace.append([main_window] + small_windows)
    #
    #     nxt_x = next_x[i][cur_x][cur_y]
    #     nxt_y = next_y[i][cur_x][cur_y]
    #     cur_x, cur_y = nxt_x, nxt_y

    recommended_trace = []
    small_window_traces = []
    for _ in range(args.num_small_windows):
        small_window_traces.append([])
    for i in range(num_frames):
        if i % key_frame_interval == 0:
            for wid in range(1 + args.num_small_windows):
                (x, y) = keyframe_trace[i // key_frame_interval][wid]
                (theta, phi) = xy2lonlat(x, y, sal_h, sal_w)
                if wid == 0:
                    recommended_trace.append((theta, phi))
                else:
                    small_window_traces[wid - 1].append((theta, phi))
        else:
            prev_key_frame = (i // key_frame_interval)
            next_key_frame = prev_key_frame + 1
            if next_key_frame >= num_key_frames:
                for wid in range(1 + args.num_small_windows):
                    (x, y) = keyframe_trace[prev_key_frame][wid]
                    (theta, phi) = xy2lonlat(x, y, sal_h, sal_w)
                    if wid == 0:
                        recommended_trace.append((theta, phi))
                    else:
                        small_window_traces[wid - 1].append((theta, phi))
            else:
                for wid in range(1 + args.num_small_windows):
                    (x0, y0) = keyframe_trace[prev_key_frame][wid]
                    (x1, y1) = keyframe_trace[next_key_frame][wid]
                    (p_theta, p_phi) = xy2lonlat(x0, y0, sal_h, sal_w)
                    (n_theta, n_phi) = xy2lonlat(x1, y1, sal_h, sal_w)
                    if wid == 0:
                        positition = 1.0 * (i % key_frame_interval) / key_frame_interval
                        theta = (1.0 - positition) * p_theta + positition * n_theta
                        phi = (1.0 - positition) * p_phi + positition * n_phi
                        recommended_trace.append((theta, phi))
                    else:
                        if p_theta == -1:
                            small_window_traces[wid - 1].append((-1, -1))
                        elif n_theta == -1:
                            small_window_traces[wid - 1].append((p_theta, p_phi))
                        else:
                            positition = 1.0 * (i % key_frame_interval) / key_frame_interval
                            theta = (1.0 - positition) * p_theta + positition * n_theta
                            phi = (1.0 - positition) * p_phi + positition * n_phi
                            small_window_traces[wid - 1].append((theta, phi))

    demo_filename_template = 'recommend_demo_{0}.mp4'
    out = cv2.VideoWriter(demo_filename_template.format(args.video_played),
                          cv2.VideoWriter_fourcc(*'DIVX'), 30, (w + w // args.num_small_windows, h))
    for i in range(num_frames):
        output = np.zeros((h, w + w // args.num_small_windows, 3)).astype('uint8')
        equ = E2P.Equirectangular(frame_list[i])
        (theta, phi) = recommended_trace[i]
        img = equ.GetPerspective(args.fov_span, theta, phi, h, w)
        output[:h, :w, ] = img
        for wid in range(args.num_small_windows):
            (theta, phi) = small_window_traces[wid][i]
            if theta == -1:
                break
            img = equ.GetPerspective(args.fov_span, theta, phi,
                                     h // args.num_small_windows, w // args.num_small_windows)
            start_x = wid * (h // args.num_small_windows)
            output[start_x: start_x + h // args.num_small_windows, w:, ] = img
        out.write(output)

        if i % 100 == 0:
            template = 'finishes output recommended demo for {0}/{1} frames'
            print(template.format(i, num_frames))


global_view_trajectories = readin_trajectories(args)
pixel2fov = E2P.Pixel2FoV(args.sal_h, args.sal_w)
saliency_maps = saliency_build(global_view_trajectories, args.video_played,
                               args.frame_path)
score, seg_score_storyline, seg_score_best, \
next_x, next_y, seg_next_x, seg_next_y, num_frames\
    = scanpath_generating(saliency_maps, pixel2fov)
play_back(score, seg_score_storyline, seg_score_best, next_x, next_y,
          seg_next_x, seg_next_y, num_frames)
