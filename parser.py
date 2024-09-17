import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--data_set', type=str, default='user_study')   # 'user_study' or 'public'
parser.add_argument('--frame_path', type=str, default='/data/yinjie/frames')
parser.add_argument('--sal_path', type=str, default='/data/yinjie/saliency_maps')
# parser.add_argument('--frame_path', type=str, default='frames')
parser.add_argument('--sal_gt_path', type=str, default='../../Pano2Vid/GroundTruths')
parser.add_argument('--ft_gt_path', type=str, default='../UserStudyDataset/gt_first_time')
parser.add_argument('--gv_gt_path', type=str, default='../UserStudyDataset/gt_global_view')
parser.add_argument('--user_trace_file', type=str, default='table1.json')
# parser.add_argument('--user_trace_file', type=str, default='../monet360video_database.json')
parser.add_argument('--fov_output_dir', type=str, default='fov_cuts')
parser.add_argument('--video_length', type=int, default=90)
parser.add_argument('--public_frame_path', type=str, default='/data/yinjie/Pano2Vid/frames')
parser.add_argument('--public_gt_path', type=str, default='/data/yinjie/Pano2Vid/ground_truths')
parser.add_argument('--human_edit_file', type=str, default='/data/yinjie/Pano2Vid/human_edit.json')
parser.add_argument('--bandwidth_trace_files', type=str, default='./bandwidth/traces_oboe')
parser.add_argument('--fov_span', type=float, default=65.5)

parser.add_argument('--gaussian_var', type=float, default=20.0)
parser.add_argument('--video_played', type=int, default=1)
parser.add_argument('--key_frame_interval', type=int, default=5)
parser.add_argument('--sal_h', type=int, default=45)
parser.add_argument('--sal_w', type=int, default=80)
parser.add_argument('--alpha0', type=float, default=0.0)
parser.add_argument('--alpha1', type=float, default=15.0)
parser.add_argument('--alpha2', type=float, default=100.0)
parser.add_argument('--gamma', type=float, default=0.95)
parser.add_argument('--beta', type=float, default=0.5)
parser.add_argument('--num_small_windows', type=float, default=2)

parser.add_argument('--fps', type=float, default=30.0)
parser.add_argument('--interval', type=int, default=4)

parser.add_argument('--min_x', type=float, default=0.16)
parser.add_argument('--max_x', type=float, default=0.80)
parser.add_argument('--min_y', type=float, default=0.14)
parser.add_argument('--max_y', type=float, default=0.98)

parser.add_argument('--anamode', type=int, default=0)

parser.add_argument('--following_data', type=str, default='following_data')
parser.add_argument('--vp_data', type=str, default='vp_data')
parser.add_argument('--log_dir', type=str, default='training_logs')
parser.add_argument('--model_dir', type=str, default='saved_models')
parser.add_argument('--input_dim', type=int, default=8)
parser.add_argument('--hidden_dim', type=int, default=16)
parser.add_argument('--output_dim', type=int, default=2)
parser.add_argument('--num_epochs', type=int, default=300)
parser.add_argument('--dropout', type=float, default=0.7)
parser.add_argument('--threshold', type=float, default=0.2)
parser.add_argument('--lr', type=float, default=0.01)
parser.add_argument('--skips', type=int, default=1)
parser.add_argument('--history_window', type=int, default=5)
parser.add_argument('--vp_method', type=str, default='LR')

parser.add_argument('--tile_h', type=int, default=4)
parser.add_argument('--tile_w', type=int, default=6)
parser.add_argument('--quality_levels', type=list, default=['0', '360P', '480P', '720P', '1K', '4K'])
parser.add_argument('--temp_sizes', type=list, default=[0, 86400., 153600., 460800., 1036800., 4147200.]) # bytes
# parser.add_argument('--predict_pos', type=int, default=1)

parser.add_argument('--ms_in_s', type=float, default=1000.0)
parser.add_argument('--b_in_mb', type=float, default=1000000.0)
parser.add_argument('--m_in_k', type=float, default=1000.0)
parser.add_argument('--bits_in_byte', type=float, default=8.0)
parser.add_argument('--video_chunk_len', type=float, default=2000.0) # in ms
parser.add_argument('--rtt', type=float, default=80.0) # in ms
parser.add_argument('--buffer_threshold', type=float, default=4000.0) # in ms
parser.add_argument('--p_rebuf', type=float, default=0.1) # in ms

args = parser.parse_args()
