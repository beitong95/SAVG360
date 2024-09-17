import os
import subprocess

video_list = ["8lsB-P8nGSM", "aQd41nbQM-U", "LKWXHKFCMO8",
              "yVLfEHXQk08", "MXlHCTXtcNs", "p9h3ZqJa1iA",
              "kiP5vWqPryY", "jMyDqZe0z7M", "g6w6xkQeSHg",
              "_7e9ej3CbuE","b6QJlGfovCU","8u3R3D1Y7Ck",
              "AYBd0oKVmhQ","nT5ete4d8v0","C7cxIamvRR4",
              "i9SiIyCyRM0", "L66XI9vIDA4","itS1LuDznDQ",
              "c5Amy1SZ1Qo","FVxqi0sAZxA", "IGofe5OL0Go",
              "Em-_GxhCPW4","Ss8zC_FjVBo"]

os.chdir('../../gpac_trying')
# os.chdir('360P')
#
# template = "youtube-dl --user-agent URL https://www.youtube.com/watch?v={0} -o {1}.mp4 -f 134+140"
# for i in range(len(video_list)):
#     print('Downloading 360P video:', i + 1)
#     video = video_list[i]
#     cmd = template.format(video, i+1)
#     output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()

# os.chdir('480P')
#
# template = "youtube-dl --user-agent URL https://www.youtube.com/watch?v={0} -o {1}.mp4 -f 135+140"
# for i in range(19, len(video_list)):
#     print('Downloading 480P video:', i + 1)
#     video = video_list[i]
#     cmd = template.format(video, i+1)
#     output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
#
# os.chdir('720P')
#
# template = "youtube-dl --user-agent URL https://www.youtube.com/watch?v={0} -o {1}.mp4 -f 136+140"
# for i in range(len(video_list)):
#     print('Downloading 720P video:', i + 1)
#     video = video_list[i]
#     cmd = template.format(video, i+1)
#     output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()

os.chdir('1K')

template = "youtube-dl --user-agent URL https://www.youtube.com/watch?v={0} -o {1}.mp4 -f 137+140"
for i in range(22, len(video_list)):
    print('Downloading 1K video:', i + 1)
    video = video_list[i]
    cmd = template.format(video, i+1)
    output = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE).stdout.read()
