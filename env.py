import numpy as np
import random
from parser import args


class Environment:
    def __init__(self, time, bw, num_chunks, video_sizes, qualities,
                 quality_levels=args.quality_levels,
                 num_tiles=args.tile_h*args.tile_w, packet_payload_portion=0.95):
        assert len(time) == len(bw)
        self.time = time
        self.bw = bw

        self.video_chunk_counter = 0
        self.buffer_size = 0
        self.mahimahi_ptr = 1
        self.last_mahimahi_time = self.time[self.mahimahi_ptr - 1]

        self.num_chunks = num_chunks
        self.num_tiles = num_tiles
        self.quality_levels = quality_levels

        self.video_size = np.zeros((num_chunks, num_tiles, len(self.quality_levels)))  # in bytes
        self.video_quality = np.zeros((num_chunks, num_tiles, len(self.quality_levels))) # psnr

        self.packet_payload_portion = packet_payload_portion

        self.get_video_data(video_sizes, qualities)

    def get_video_data(self, video_sizes, qualities):
        # load json
        for chunk_no in range(self.num_chunks):
            for tile_no in range(self.num_tiles):
                for level_no in range(len(self.quality_levels)):
                    quality_level = self.quality_levels[level_no]
                    if level_no == 0:
                        self.video_size[chunk_no, tile_no, level_no] = 1
                    else:
                        self.video_size[chunk_no, tile_no, level_no] = \
                            video_sizes[(chunk_no, quality_level, tile_no)]
                        self.video_quality[chunk_no, tile_no, level_no] = \
                            qualities[(chunk_no, quality_level, tile_no)]

    def get_video_chunk(self, selected_levels):

        video_chunk_size = np.float64(0)
        for tile_no in range(len(selected_levels)):
            quality_level = selected_levels[tile_no]
            video_chunk_size += np.float64(self.video_size[self.video_chunk_counter,
                                                           tile_no, quality_level])
        # use the delivery opportunity in mahimahi
        delay = 0.0  # in ms
        video_chunk_counter_sent = 0  # in bytes
        while True:  # download video chunk over mahimahi
            throughput = self.bw[self.mahimahi_ptr] * args.b_in_mb / args.bits_in_byte
            duration = self.time[self.mahimahi_ptr] - self.last_mahimahi_time
            packet_payload = throughput * duration * self.packet_payload_portion
            if video_chunk_counter_sent + packet_payload > video_chunk_size:

                fractional_time = (video_chunk_size - video_chunk_counter_sent) / \
                                  throughput / self.packet_payload_portion
                delay += fractional_time
                self.last_mahimahi_time += fractional_time
                break

            video_chunk_counter_sent += packet_payload
            delay += duration
            self.last_mahimahi_time = self.time[self.mahimahi_ptr]
            self.mahimahi_ptr += 1

            if self.mahimahi_ptr >= len(self.bw):
                # loop back in the beginning
                # note: trace file starts with time 0
                self.mahimahi_ptr = 1
                self.last_mahimahi_time = 0
        delay *= args.ms_in_s
        delay += args.rtt

        # rebuffer time
        rebuf = np.maximum(delay - self.buffer_size, 0.0)
        # print(delay / MILLISECONDS_IN_SECOND, self.buffer_size / MILLISECONDS_IN_SECOND)

        # update the buffer
        self.buffer_size = np.maximum(self.buffer_size - delay, 0.0)

        # add in the new chunk
        self.buffer_size += args.video_chunk_len

        # sleep if buffer gets too large
        sleep_time = 0
        if self.buffer_size > args.buffer_threshold:
            # exceed the buffer limit
            # we need to skip some network bandwidth here
            # but do not add up the delay
            drain_buffer_time = self.buffer_size - args.buffer_threshold
            sleep_time = drain_buffer_time
            self.buffer_size -= sleep_time
            while True:
                duration = self.time[self.mahimahi_ptr]  - self.last_mahimahi_time
                if duration > sleep_time / args.ms_in_s:
                    self.last_mahimahi_time += sleep_time / args.ms_in_s
                    break
                sleep_time -= duration * args.ms_in_s
                self.last_mahimahi_time = self.time[self.mahimahi_ptr]
                self.mahimahi_ptr += 1

                if self.mahimahi_ptr >= len(self.bw):
                    # loop back in the beginning
                    # note: trace file starts with time 0
                    self.mahimahi_ptr = 1
                    self.last_mahimahi_time = 0
        # the "last buffer size" return to the controller
        # Note: in old version of dash the lowest buffer is 0.
        # In the new version the buffer always have at least
        # one chunk of video
        return_buffer_size = self.buffer_size

        this_chunk_info = (self.video_size[self.video_chunk_counter],
                                 self.video_quality[self.video_chunk_counter])

        self.video_chunk_counter += 1
        video_chunk_remain = self.num_chunks - self.video_chunk_counter

        end_of_video = False
        if self.video_chunk_counter >= self.num_chunks:
            end_of_video = True
            self.buffer_size = 0
            self.video_chunk_counter = 0
            self.mahimahi_ptr = 1
            self.last_mahimahi_time = self.time[self.mahimahi_ptr - 1]

        next_chunk_info = (self.video_size[self.video_chunk_counter],
                           self.video_quality[self.video_chunk_counter])
        return delay, \
            sleep_time, \
            return_buffer_size / args.ms_in_s, \
            rebuf / args.ms_in_s, \
            video_chunk_size, \
            this_chunk_info, \
            next_chunk_info, \
            end_of_video, \
            video_chunk_remain
