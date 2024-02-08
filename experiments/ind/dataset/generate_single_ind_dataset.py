import argparse
import os

import torch

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Build ind datasets')
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', required=True)
    parser.add_argument('--scenes_to_use', type=int, nargs='+', default=[2])
    args = parser.parse_args()

    train_path = os.path.join(args.data_dir, 'processed_train_data')
    train_feats, train_masks = torch.load(train_path)
    val_path = os.path.join(args.data_dir, 'processed_val_data')
    val_feats, val_masks = torch.load(val_path)
    test_path = os.path.join(args.data_dir, 'processed_test_data')
    test_feats, test_masks = torch.load(test_path)

    all_feats = sum([train_feats, val_feats, test_feats], [])
    all_masks = sum([train_masks, val_masks, test_masks], [])

    scene_indices = [
        slice(7),
        slice(7, 18),
        slice(18, 30),
        slice(30, 33)
    ]
    train_indices = [
        slice(0, 5),
        slice(7, 14),
        slice(18, 26),
        slice(30, 31)
    ]
    val_indices = [
        slice(5, 6),
        slice(14, 16),
        slice(26, 28),
        slice(31, 32)
    ]
    test_indices = [
        slice(6, 7),
        slice(16, 18),
        slice(28, 30),
        slice(32, 33)
    ]

    scene_indices = [scene_indices[idx] for idx in args.scenes_to_use]
    train_indices = [train_indices[idx] for idx in args.scenes_to_use]
    val_indices = [val_indices[idx] for idx in args.scenes_to_use]
    test_indices = [test_indices[idx] for idx in args.scenes_to_use]

    num_full_steps = 75
    new_feats = []
    new_masks = []
    for indices in scene_indices:
        new_feats.append(sum([
            list(feat.split(num_full_steps))[:-1] for feat in all_feats[indices]], []))
        new_masks.append(sum([
            list(mask.split(num_full_steps))[:-1] for mask in all_masks[indices]], []))

    new_train_feats = []
    new_train_masks = []
    for indices in train_indices:
        new_train_feats.append(sum([
            list(feat.split(num_full_steps))[:-1] for feat in all_feats[indices]], []))
        new_train_masks.append(sum([
            list(mask.split(num_full_steps))[:-1] for mask in all_masks[indices]], []))

    new_val_feats = []
    new_val_masks = []
    for indices in val_indices:
        new_val_feats.append(sum([
            list(feat.split(num_full_steps))[:-1] for feat in all_feats[indices]], []))
        new_val_masks.append(sum([
            list(mask.split(num_full_steps))[:-1] for mask in all_masks[indices]], []))

    new_test_feats = []
    new_test_masks = []
    for indices in test_indices:
        new_test_feats.append(sum([
            list(feat.split(num_full_steps))[:-1] for feat in all_feats[indices]], []))
        new_test_masks.append(sum([
            list(mask.split(num_full_steps))[:-1] for mask in all_masks[indices]], []))

    for idx in range(len(new_feats)):
        out_path = os.path.join(args.output_dir, f'single_ind_processed_{args.scenes_to_use[idx]}')
        if not os.path.exists(out_path):
            os.makedirs(out_path)

        train_path = os.path.join(out_path, 'processed_train_data')
        torch.save((new_train_feats[idx], new_train_masks[idx]), train_path)
        val_path = os.path.join(out_path, 'processed_val_data')
        torch.save((new_val_feats[idx], new_val_masks[idx]), val_path)
        test_path = os.path.join(out_path, 'processed_test_data')
        torch.save((new_test_feats[idx], new_test_masks[idx]), test_path)
