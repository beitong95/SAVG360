import os
import subprocess
from parser import args

video_begin = [20, 30, 48, 0, 1, 10, 10, 2]

os.chdir('../../gpac_trying')
num_videos = 23
bitrate = ['360P', '480P', '720P', '1K']
width_height = [(640, 360), (854, 480), (1280, 720), (1920, 1080)]

def time_crop(bitrate_level):
    input_video_template = '{0}/{1}.mp4'
    output_video_template = '{0}/{1}_{2}.mp4'
    cmd_template = 'ffmpeg -ss {0} -i {1} -c copy -t {2} {3}'
    for i in range(num_videos):
        input_video = input_video_template.format(bitrate[bitrate_level], i + 1)
        output_video = output_video_template.format('videos', i + 1, bitrate[bitrate_level])
        start_time, video_length = 0, 90
        if i < len(video_begin):
            start_time, video_length = video_begin[i], 60
        cmd = cmd_template.format(start_time, input_video, video_length, output_video)
        output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()


def tile_crop(video, bitrate_level):
    if not os.path.exists("tiled_videos"):
        os.mkdir("tiled_videos")
    if not os.path.exists("tiled_videos/{0}".format(video)):
        os.mkdir("tiled_videos/{0}".format(video))
    cmd = "ffmpeg -i videos/{0}_{1}.mp4 -filter_complex ".format(video, bitrate[bitrate_level])
    split_header = "[0]split={0}".format(args.tile_h * args.tile_w)
    split_cmd = ""
    split_template = "; [s{0}]crop={1}:{2}:{3}:{4}[s{0}]"
    map_cmd = ""
    map_template = " -map [s{0}] -c:v libx264 tiled_videos/{1}/tile_{2}_{3}_bitrate_{4}.mp4"
    tile_width = width_height[bitrate_level][0] / args.tile_w
    tile_height = width_height[bitrate_level][1] / args.tile_h
    tile_id = 0
    for i in range(args.tile_h):
        for j in range(args.tile_w):
            x = int(tile_width * j)
            y = int(tile_height * i)
            w = int(tile_width * (j + 1)) - x
            h = int(tile_height * (i + 1)) - y
            split_header += '[s{0}]'.format(tile_id)
            split_cmd += split_template.format(tile_id, w, h, x, y)
            map_cmd += map_template.format(tile_id, video, i, j, bitrate_level)
            tile_id = tile_id + 1
            # time_crop(3)
    cmd = cmd + '"' + split_header + split_cmd + '"' + map_cmd
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()

for video in range(num_videos):
    for bitrate_level in range(len(bitrate)):
        tile_crop(video+1, bitrate_level)