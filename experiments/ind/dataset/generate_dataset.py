import os
import argparse

import numpy as np
import torch

import experiments.ind.dataset.ind_data_utils as idu


if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build ind datasets')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--num_train', type=int, default=19)
    parser.add_argument('--num_val', type=int, default=7)
    parser.add_argument('--downsample_factor', type=int, default=10)
    args = parser.parse_args()

    all_tracks, all_static, all_meta = idu.read_all_recordings_from_csv(args.data_dir)
    all_feats = []
    all_masks = []
    min_feats = np.array([100000000000000, 100000000000000, 100000000000000, 10000000000000])
    max_feats = np.array([-100000000000000, -100000000000000, -100000000000000, -10000000000000])
    for track_set_id, track_set in enumerate(all_tracks):
        num_tracks = len(track_set)
        max_frame = 0
        for track_info in all_static[track_set_id]:
            max_frame = max(max_frame, track_info['finalFrame'])
        print("%d: %d" % (track_set_id, max_frame))
        feats = np.zeros((max_frame+1, num_tracks, 4))
        masks = np.zeros((max_frame+1, num_tracks))
        for track_id, track in enumerate(track_set):
            frames = track['frame']
            feats[frames, track_id, 0] = track['xCenter']
            feats[frames, track_id, 1] = track['yCenter']
            feats[frames, track_id, 2] = track['xVelocity']
            feats[frames, track_id, 3] = track['yVelocity']
            masks[frames, track_id] = 1
            if track_set_id < args.num_train:
                min_feats[0] = min(min_feats[0], track['xCenter'].min())
                min_feats[1] = min(min_feats[1], track['yCenter'].min())
                min_feats[2] = min(min_feats[2], track['xVelocity'].min())
                min_feats[3] = min(min_feats[3], track['yVelocity'].min())
                max_feats[0] = max(max_feats[0], track['xCenter'].max())
                max_feats[1] = max(max_feats[1], track['yCenter'].max())
                max_feats[2] = max(max_feats[2], track['xVelocity'].max())
                max_feats[3] = max(max_feats[3], track['yVelocity'].max())
        all_feats.append(torch.FloatTensor(feats[::args.downsample_factor]))
        all_masks.append(torch.FloatTensor(masks[::args.downsample_factor]))
    train_feats = all_feats[:args.num_train]
    val_feats = all_feats[args.num_train:args.num_train+args.num_val]
    test_feats = all_feats[args.num_train+args.num_val:]
    train_masks = all_masks[:args.num_train]
    val_masks = all_masks[args.num_train:args.num_train+args.num_val]
    test_masks = all_masks[args.num_train+args.num_val:]

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    train_path = os.path.join(args.output_dir, 'processed_train_data')
    torch.save([train_feats, train_masks], train_path)
    val_path = os.path.join(args.output_dir, 'processed_val_data')
    torch.save([val_feats, val_masks], val_path)
    test_path = os.path.join(args.output_dir, 'processed_test_data')
    torch.save([test_feats, test_masks], test_path)

    stats_path = os.path.join(args.output_dir, 'train_data_stats')
    torch.save([torch.FloatTensor(min_feats), torch.FloatTensor(max_feats)], stats_path)

    flattened_train_feats = torch.cat([xi.reshape(-1, 4) for xi in train_feats], 0)
    flattened_train_masks = torch.cat([xi.reshape(-1) for xi in train_masks], 0)
    max_speed = torch.norm(flattened_train_feats[flattened_train_masks.bool()][:, 2:], dim=-1).max()

    speed_norm_path = os.path.join(args.output_dir, 'train_speed_norm_stats')
    torch.save(max_speed, os.path.join(args.output_dir, speed_norm_path))
