import cv2
import json
import numpy as np
from scipy import stats
from parser import args


def harmonic_avg(past_bandwidths):
    n, tot = 0, 0
    for bw in past_bandwidths:
        if bw != 0:
            n += 1
            tot += 1.0 / bw
    if n == 0:
        return 0
    return n / tot


def xy2lonlat(x, y, h, w):
    if x == -1:
        return (-1, -1)
    phi = - ((x + 0.5) / h - 0.5) * 180
    theta = ((y + 0.5) / w - 0.5) * 360
    return (theta, phi)


def lonlat2xyz(theta, phi):
    x_axis = np.array([1.0, 0.0, 0.0], np.float32)
    y_axis = np.array([0.0, 1.0, 0.0], np.float32)
    z_axis = np.array([0.0, 0.0, 1.0], np.float32)
    R1, _ = cv2.Rodrigues(y_axis * np.radians(theta))
    R2, _ = cv2.Rodrigues(np.dot(R1, x_axis) * np.radians(phi))
    R = R2 @ R1
    xyz = z_axis @ R.T
    return xyz


def lonlat2pos(theta, phi):
    x = round((0.5 - phi) * args.sal_h - 0.5)
    y = round((0.5 + theta) * args.sal_w - 0.5)
    return (x, y), x * args.sal_w + y


def lonlat2dist(theta0, phi0, theta1, phi1):
    xyz0 = lonlat2xyz(theta0 * 360, phi0 * 180)
    xyz1 = lonlat2xyz(theta1 * 360, phi1 * 180)
    dist = np.sum((xyz0 - xyz1) ** 2)
    return dist


memory_dist = {}

def pixel_dist(x0, y0, x1, y1):
    if (x0, y0, x1, y1) in memory_dist.keys():
        return memory_dist[(x0, y0, x1, y1)]
    (theta, phi) = xy2lonlat(x0, y0, args.sal_h, args.sal_w)
    xyz0 = lonlat2xyz(theta, phi)
    (theta, phi) = xy2lonlat(x1, y1, args.sal_h, args.sal_w)
    xyz1 = lonlat2xyz(theta, phi)
    dist = np.sum((xyz0 - xyz1) ** 2)
    memory_dist[(x0, y0, x1, y1)] = dist
    return dist

def get_deg_distance(deg1, deg2):
    (theta1, phi1) = deg1
    v1 = lonlat2xyz(theta1, phi1)
    (theta2, phi2) = deg2
    v2 = lonlat2xyz(theta2, phi2)
    return np.arccos(np.dot(v1, v2)) / np.pi * 180


_gaussian_dict = {
    np.around(_d, 1):
        stats.multivariate_normal.pdf(_d, mean=0, cov=args.gaussian_var)
    for _d in np.arange(0.0, 180, .1)}


def gaussian_from_distance(_d):
    temp = np.around(_d, 1)
    return _gaussian_dict[temp] if temp in _gaussian_dict else 0.0


def readin_trajectories(args):
    with open(args.user_trace_file, 'r') as f:
        data = json.load(f)
        for x in data:
            if x['type'] == "table":
                trajectory_data = x['data']

    cleaned_data = []
    for data_point in trajectory_data:
        if data_point['username'].find("qian") != -1:
            continue
        if data_point['username'].find("Klara124431") != -1:
            continue
        cleaned_data.append(data_point)

    # first_time_trajs = {}
    global_view_trajs = {}
    for data_point in cleaned_data:
        x = data_point['x'].lstrip('[').rstrip(']').split(',')
        y = data_point['y'].lstrip('[').rstrip(']').split(',')
        x = [float(str) for str in x]
        y = [float(str) for str in y]

        videoId = int(data_point['video'])
        # videoName = 'v' + str(videoId + 1)
        if data_point['step'] == "2":
            if videoId not in global_view_trajs.keys():
                global_view_trajs[videoId] = []
            global_view_trajs[videoId].append((x, y))

    return global_view_trajs


def loc_interpolate(x, y, i, args):
    if i == 0:
        return 0.0, 0.0
    seg_begin = i
    while x[seg_begin] == 999:
        seg_begin = seg_begin - 1
    seg_end = i
    while seg_end < len(x) and x[seg_end] == 999:
        seg_end = seg_end + 1
    if seg_end == len(x):
        return x[seg_begin], y[seg_begin]
    else:
        # print(i, seg_begin, seg_end)
        seg_frame_interval = seg_end - seg_begin
        seg_pos = (i % seg_frame_interval) / seg_frame_interval
        s_theta, s_phi = x[seg_begin], y[seg_begin]
        e_theta, e_phi = x[seg_end], y[seg_end]
        d_theta = e_theta - s_theta
        d_phi = e_phi - s_phi
        theta = s_theta + seg_pos * d_theta
        phi = s_phi + seg_pos * d_phi
        return theta, phi