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
        self.traj = traj
        self.guidance_data = self.guidance_readin()
        # guidance_data : {"fps"; "key_frame_interval"; "num_key_frames";
        #                  "sal_h"; "sal_w"; "segment_ref";"segment_info";
        #                  "global_guidance";"segment_guidance";"start_pos"}
        self.labels = np.zeros((self.guidance_data["num_key_frames"] * 5, 2))
        # self.estimated_time = None
        self.sal_h, self.sal_w = args.sal_h, args.sal_w

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


    def vp_seq(self):
        frame_interval = self.guidance_data["key_frame_interval"] // 5
        for i in range(self.guidance_data["num_key_frames"] * 5):
            frame_idx = min(len(self.traj) - 1, i * frame_interval)
            (x, y) = self.get_xy(traj[frame_idx])
            (theta, phi) = xy2lonlat(x, y, self.sal_h, self.sal_w)
            self.labels[i][0] = theta / 360
            self.labels[i][1] = phi / 180
        return self.labels


if __name__ == '__main__':
    data = readin()
    template = 'vp_data' + os.sep + '{0}_{1}.json'
    for trace in data[1:]:
        username, video_id, mode, str_traj = trace[1][:5], int(trace[2]), int(trace[3]), trace[4]
        if mode == 0 or (username[0] >= '0' and username[0] <= '9'):
            continue
        traj = get_traj(str_traj)
        client = Client(video_id, traj)
        labels = client.vp_seq()
        np.save(template.format(username, video_id), labels)
