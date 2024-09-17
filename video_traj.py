import os
import glob
import cv2
from utils import *
import Equirec2Perspec as E2P


def fov_video_output(trajs, video_name, input_path=args.public_frame_path):
    frame_dir = input_path + os.sep + str(video_name + 1)
    frame_list = sorted(glob.glob(frame_dir + os.sep + '*'))
    h, w = 360, 640
    if not os.path.exists(frame_dir):
        return

    demo_filename_template = 'fov_demo_{0}.mp4'
    out = cv2.VideoWriter(demo_filename_template.format(video_name),
                          cv2.VideoWriter_fourcc(*'DIVX'), 30, (w, h))
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
        (theta, phi) = hdmv_points[2]
        equ = E2P.Equirectangular(frame_list[i])
        img = equ.GetPerspective(args.fov_span, theta, phi, h, w)
        out.write(img)
        if i % 100 == 0:
            template = 'video {0} writing {1}/{2} frames'
            print(template.format(video_name, i, num_frames))


global_view_trajectories = readin_trajectories(args)
fov_video_output(global_view_trajectories, args.video_played, args.frame_path)
