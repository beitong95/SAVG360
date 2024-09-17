import csv
import os
import json
from parser import args
import glob
from utils import *
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams

def readin():
    data = []
    with open('rating2.csv', 'r') as f:
        f_csv = csv.reader(f, delimiter=';')
        for row in f_csv:
            data.append(row)
    return data


def get_traj(str_traj):
    traj = []
    num_pixels = args.sal_w * args.sal_h
    for i in str_traj.lstrip('[').rstrip(']').split(','):
        val = eval(i)
        if val == -1:
            traj.append(traj[-1])
        else:
            traj.append(val % num_pixels)
    return traj


class Player:
    def __init__(self, video_id, h, w, start_x=-1, start_y=-1):
        self.h, self.w = h, w
        # print(start_x, start_y)
        self.current = (start_x, start_y)
        self.next = None

    def set_current(self, viewport):
        self.current = viewport

    def set_target(self, target):
        self.next = target

    def lonlat2pixel(self, theta, phi):
        if theta == -1:
            return (-1, -1)
        x = round((0.5 - (phi / 180)) * self.h - 0.5)
        y = round((0.5 + theta / 360) * self.w - 0.5)
        return (x, y)

    def move(self):
        self.current = self.next

    def get_pos(self, position):
        (p_theta, p_phi) = xy2lonlat(self.current[0], self.current[1], self.h, self.w)
        (n_theta, n_phi) = xy2lonlat(self.next[0], self.next[1], self.h, self.w)
        theta = (1.0 - position) * p_theta + position * n_theta
        phi = (1.0 - position) * p_phi + position * n_phi
        return self.lonlat2pixel(theta, phi)


class Client:
    def __init__(self, video_id, traj):
        self.video = video_id
        self.key_frame_idx = 0
        self.segment_idx = 0
        self.traj = traj
        self.guidance_data = self.guidance_readin()
        # guidance_data : {"fps"; "key_frame_interval"; "num_key_frames";
        #                  "sal_h"; "sal_w"; "segment_ref";"segment_info";
        #                  "global_guidance";"segment_guidance";"start_pos"}
        self.main_player = None
        self.main_mark = None
        self.side_players = []
        self.side_marks = []
        self.labels = np.zeros((self.guidance_data["num_key_frames"], args.input_dim))
        # self.estimated_time = None
        self.sal_h, self.sal_w = args.sal_h, args.sal_w
        self.following, self.follow_1, self.follow_2, self.brk = 0, 0, 0, 0

    def get_pos(self, x, y):
        return x * self.sal_w + y

    def get_xy(self, pos):
        return (pos // self.sal_w, pos % self.sal_w)

    def guidance_readin(self):
        json_template = 'guidance' + os.sep + '{0}.json'
        with open(json_template.format(self.video), 'r', encoding="utf-8") as f:
            json_str = json.load(f)
            guidance_data = json.loads(json_str)
        if self.video < 9:
            guidance_data["interval"] = 3
        else:
            guidance_data["interval"] = 4
        return guidance_data

    def setup_play(self):
        (start_x, start_y) = self.get_xy(self.guidance_data["start_pos"])
        self.main_player = Player(self.video, self.sal_h, self.sal_w, start_x, start_y)
        for _ in range(args.num_small_windows):
            self.side_players.append(Player(self.video, self.sal_h, self.sal_w))
            self.side_marks.append(-1)

    def set_target(self):
        (x, y) = self.main_player.current
        target = self.guidance_data["global_guidance"][self.key_frame_idx][self.get_pos(x, y)]
        target = self.get_xy(target)
        self.main_player.set_target(target)
        for i in range(args.num_small_windows):
            (x, y) = self.side_players[i].current
            if x != -1:
                target = self.guidance_data["segment_guidance"][self.key_frame_idx][self.get_pos(x, y)]
                target = self.get_xy(target)
                self.side_players[i].set_target(target)
            else:
                self.side_players[i].set_target((-1, -1))

    def get_label(self, theta, phi):
        label = np.zeros(args.input_dim)
        label[0] = theta / 360
        label[1] = phi / 180
        total = self.following + self.follow_1 + self.follow_2 + self.brk
        label[2] = self.following / total
        if self.main_mark is not None:
            label[self.main_mark + 3] = self.following / total
        if self.side_marks[0] != -1:
            label[self.side_marks[0] + 3] = self.follow_1 / total
        if self.side_marks[1] != -1:
            label[self.side_marks[1] + 3] = self.follow_2 / total
        label[-1] = self.brk / total
        return label

    def video_play(self):
        seg_frame_interval = round((self.guidance_data["interval"] * self.guidance_data["fps"])
                                   / self.guidance_data["key_frame_interval"])
        seg_start = self.segment_idx * seg_frame_interval
        seg_end = min(seg_start + seg_frame_interval, self.guidance_data["num_key_frames"])
        # (todo) request next segment

        (start_x0, start_y0) = self.main_player.current
        start_pos = self.get_pos(start_x0, start_y0)
        score_ref = self.guidance_data["segment_ref"][self.segment_idx][start_pos]
        cur_wid = 0
        num_choices = len(self.guidance_data["segment_info"][self.segment_idx]["score"])
        self.main_mark = None
        for i in range(num_choices):
            if cur_wid >= args.num_small_windows:
                break
            score = self.guidance_data["segment_info"][self.segment_idx]["score"][i]
            if score < args.beta * score_ref:
                continue
            (x, y) = self.get_xy(self.guidance_data["segment_info"][self.segment_idx]["pos_start"][i])
            if pixel_dist(start_x0, start_y0, x, y) <= 0.01 and self.main_mark is None:
                self.main_mark = i
            if abs(start_x0 - x) < self.sal_h // 6 or \
                    self.sal_h - abs(start_x0 - x) < self.sal_h // 6:
                continue
            if abs(start_y0 - y) < self.sal_w // 6 or \
                    self.sal_w - abs(start_y0 - y) < self.sal_w // 6:
                continue
            self.side_players[cur_wid].set_current(
                self.get_xy(self.guidance_data["segment_info"][self.segment_idx]["pos_start"][i]))
            self.side_marks[cur_wid] = i
            cur_wid = cur_wid + 1
        # print(cur_wid)
        for i in range(cur_wid, args.num_small_windows):
            self.side_players[i].set_current((-1, -1))
            self.side_marks[cur_wid] = -1
        #
        assert (self.guidance_data["interval"] * self.guidance_data["fps"]) % \
               self.guidance_data["key_frame_interval"] <= 1e-3 or \
               (self.guidance_data["interval"] * self.guidance_data["fps"]) % \
               self.guidance_data["key_frame_interval"] >= self.guidance_data["key_frame_interval"] - 1e-3
        # tic = time.time()
        while self.key_frame_idx < self.guidance_data["num_key_frames"]:
            self.following, self.might_following, self.follow_1, self.follow_2, self.brk = 0, 0, 0, 0, 0
            frame_idx = min(len(self.traj) - 1, self.key_frame_idx *
                            self.guidance_data["key_frame_interval"])
            current = self.get_xy(traj[frame_idx])
            self.main_player.set_current(current)
            self.set_target()
            record = []
            for i in range(1, self.guidance_data["key_frame_interval"] + 1):
                frame_idx = min(len(self.traj) - 1, self.key_frame_idx *
                                self.guidance_data["key_frame_interval"] + i)
                viewport = traj[frame_idx]
                (x, y) = self.get_xy(viewport)
                position = 1.0 * i / self.guidance_data["key_frame_interval"]
                (x00, y00) = self.main_player.get_pos(0.0)
                (x01, y01) = self.main_player.get_pos(1.0)
                (x10, y10) = self.side_players[0].get_pos(0.0)
                (x11, y11) = self.side_players[0].get_pos(1.0)
                (x20, y20) = self.side_players[1].get_pos(0.0)
                (x21, y21) = self.side_players[1].get_pos(1.0)
                # record.append(((x, y), (x00, y00), (x01, y01)))
                if (x, y) == (x00, y00):
                    if position <= 0.5:
                        self.following += 1
                    else:
                        self.might_following += 1
                elif (x, y) == (x01, y01):
                    if position >= 0.5:
                        self.following += 1
                    else:
                        self.might_following += 1
                elif (x, y) == (x10, y10) or (x, y) == (x11, y11):
                    self.follow_1 += 1
                    pos1 = self.get_pos(self.side_players[0].current[0], self.side_players[0].current[1])
                    target = self.guidance_data["global_guidance"][self.key_frame_idx][pos1]
                    target = self.get_xy(target)
                    self.side_players[0].set_target(target)
                elif (x, y) == (x20, y20) or (x, y) == (x21, y21):
                    self.follow_2 += 1
                    pos2 = self.get_pos(self.side_players[1].current[0], self.side_players[1].current[1])
                    target = self.guidance_data["global_guidance"][self.key_frame_idx][pos2]
                    target = self.get_xy(target)
                    self.side_players[1].set_target(target)
                else:
                    self.brk += 1
            # () = current
            (orig_x, orig_y) = self.main_player.get_pos(0.0)
            orig_pos = self.get_pos(orig_x, orig_y)
            (target_x, target_y) = self.main_player.get_pos(1.0)
            target_pos = self.get_pos(target_x, target_y)
            if orig_pos == target_pos:
                print("unmove")
            else:
                print("move")

            self.main_player.move()
            for player in self.side_players:
                player.move()
            (x, y) = self.get_xy(traj[min(len(self.traj) - 1, (self.key_frame_idx + 1) *
                            self.guidance_data["key_frame_interval"])])
            real_pos = self.get_pos(x, y)
            if target_pos == real_pos:
                self.following += self.might_following
            else:
                self.brk += self.might_following
            (theta, phi) = xy2lonlat(x, y, self.sal_h, self.sal_w)
            self.labels[self.key_frame_idx] = self.get_label(theta, phi)
            # if orig_pos != target_pos and self.labels[self.key_frame_idx][2] == 1.0:
            #     print(orig_pos, target_pos)
            # if real_pos == target_pos and self.labels[self.key_frame_idx][2] != 1.0:
            #     if self.labels[self.key_frame_idx][2] >= 0.4 and \
            #             self.labels[self.key_frame_idx][2] <= 0.6:
            #         print(self.labels[self.key_frame_idx][2], record)
            self.key_frame_idx = self.key_frame_idx + 1
            if self.key_frame_idx == seg_end:
                break

    def streaming(self):
        self.setup_play()
        while self.key_frame_idx < self.guidance_data["num_key_frames"]:
            self.video_play()
            self.segment_idx = self.segment_idx + 1
        return self.labels


if __name__ == '__main__':
    data = readin()
    template = 'following_data' + os.sep + '{0}_{1}.json'
    follows, unfollows = 0, 0
    for trace in data[1:]:
        username, video_id, mode, str_traj = trace[1][:5], int(trace[2]), int(trace[3]), trace[4]
        if mode == 0 or (username[0] >= '0' and username[0] <= '9'):
            continue
        traj = get_traj(str_traj)
        client = Client(video_id, traj)
        labels = client.streaming()
        # np.save(template.format(username, video_id), labels)

