import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from parser import args
import os
import glob
import random
from utils import *
from Equirec2Perspec import get_tile_in_FoV

random.seed(1)
torch.manual_seed(1)


def load_vp_data(input_following_dir=args.following_data, input_vp_dir=args.vp_data):
    train_data, test_data = [], []
    input_files = sorted(glob.glob(input_following_dir + os.sep + '*'))
    # label_counts = np.zeros(args.output_dim)
    random.shuffle(input_files)
    training_size = round(len(input_files) * 0.8)
    file_idx = 0
    for input_following_file in input_files:
        input_vp_file = input_following_file.replace(input_following_dir, input_vp_dir)
        suffix = input_following_file.split('/')[-1].split('.')[0]
        video_id = int(suffix.split('_')[-1])
        file_idx = file_idx + 1
        following_data = np.load(input_following_file)
        vp_data = np.load(input_vp_file)
        seq_len = following_data.shape[0]
        if file_idx <= training_size:
            train_data.append({"video_id": video_id, "vp_data": []})
            for i in range(seq_len):
                train_data[-1]["vp_data"].append(vp_data[i * 5: (i + 1) * 5 + 1, :])
        else:
            test_data.append({"video_id": video_id, "vp_data": []})
            for i in range(seq_len):
                test_data[-1]["vp_data"].append(vp_data[i * 5: (i + 1) * 5 + 1, :])
    # print(len(train_data), len(test_data))
    # for vp_trace in test_data:
    #     print(len(vp_trace["data"]))
    return train_data, test_data


class Navigation_Graph:
    def __init__(self, tile_h, tile_w):
        self.tile_h = tile_h
        self.tile_w = tile_w
        self.num_tiles = tile_h * tile_w
        self.views = {}
        self.view2no = {}
        self.view_counts = {}
        self.graphs = {}
        self.dir = 'navigation_graphs'

    def get_tile_set(self, chunk_vp):
        tile_vp = np.ones((self.tile_h, self.tile_w))
        for _ in chunk_vp:
            vp = (_[0].item() * 360, _[1].item() * 180)
            tile_vp = tile_vp * (1 - get_tile_in_FoV(vp))
        tile_vp = 1 - tile_vp
        tile_vp = np.reshape(tile_vp, self.num_tiles)
        tile_set = ()
        for chunk_no in range(self.num_tiles):
            if tile_vp[chunk_no] == 1.0:
                tile_set += (chunk_no,)
        return tile_set

    def add_views(self, video_id, vp_data):
        if video_id not in self.views:
            self.views[video_id] = []
            self.view2no[video_id] = {}
            self.view_counts[video_id] = {}
        for chunk_vp in vp_data:
            tile_set = self.get_tile_set(chunk_vp)
            if tile_set not in self.view2no[video_id]:
                self.view2no[video_id][tile_set] = len(self.views[video_id])
                self.views[video_id].append(tile_set)
                self.view_counts[video_id][tile_set] = 0
            self.view_counts[video_id][tile_set] += 1

    def init_graph(self):
        for video_id in self.views:
            num_views = len(self.views[video_id])
            self.graphs[video_id] = np.zeros((num_views, num_views))

    def build_graph(self, video_id, vp_data):
        num_chunks = len(vp_data)
        for chunk_no in range(num_chunks):
            tile_set_from = self.get_tile_set(vp_data[chunk_no])
            tile_set_to = tile_set_from
            if chunk_no < num_chunks - 1:
                tile_set_to = self.get_tile_set(vp_data[chunk_no + 1])
            view_from = self.view2no[video_id][tile_set_from]
            view_to = self.view2no[video_id][tile_set_to]
            self.graphs[video_id][view_from][view_to] += \
                1.0 / self.view_counts[video_id][tile_set_from]

    def store_graph(self):
        for video_id in self.views:
            view_path = self.dir + os.sep + str(video_id) + '_views'
            with open(view_path, 'w') as f:
                for view in self.views[video_id]:
                    line = str(view[0])
                    for tile in view[1: ]:
                        line = line + ' ' + str(tile)
                    print(line, file=f)
            graph_path = self.dir + os.sep + str(video_id) + '_graph.npy'
            with open(graph_path, 'wb') as f:
                np.save(f, self.graphs[video_id])


if __name__ == '__main__':
    train_data, test_data = load_vp_data()
    navigation_graph = Navigation_Graph(args.tile_h, args.tile_w)
    for vp_trace in train_data:
        navigation_graph.add_views(vp_trace["video_id"], vp_trace["vp_data"])
    navigation_graph.init_graph()
    for vp_trace in train_data:
        navigation_graph.build_graph(vp_trace["video_id"], vp_trace["vp_data"])
    navigation_graph.store_graph()
