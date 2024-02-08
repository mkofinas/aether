import os
import pickle
import torch
import numpy as np

data_dir = 'dataset/data/gravitational_field_3d'
simulation = 'gravitational_field'
ndim = 3
num_particles = 5
with open(os.path.join(data_dir, f'ds_train_{simulation}_{ndim}D_{num_particles}_new.pkl'), 'rb') as f:
    train_data = pickle.load(f)

with open(os.path.join(data_dir, f'ds_valid_{simulation}_{ndim}D_{num_particles}_new.pkl'), 'rb') as f:
    valid_data = pickle.load(f)

with open(os.path.join(data_dir, f'ds_test_{simulation}_{ndim}D_{num_particles}_new.pkl'), 'rb') as f:
    test_data = pickle.load(f)

train_feats = np.concatenate([train_data['points'], train_data['vel']], -1)
valid_feats = np.concatenate([valid_data['points'], valid_data['vel']], -1)
test_feats = np.concatenate([test_data['points'], test_data['vel']], -1)

train_feats, train_field = np.split(train_feats, [num_particles], axis=2)
valid_feats, valid_field = np.split(valid_feats, [num_particles], axis=2)
test_feats, test_field = np.split(test_feats, [num_particles], axis=2)

train_field = train_field[:, 0]
valid_field = valid_field[:, 0]
test_field = test_field[:, 0]

train_feats = train_feats[:, 1:]
valid_feats = valid_feats[:, 1:]
test_feats = test_feats[:, 1:]

torch.save(torch.from_numpy(train_feats).float(), os.path.join(data_dir, 'train_feats'))
torch.save(torch.from_numpy(valid_feats).float(), os.path.join(data_dir, 'valid_feats'))
torch.save(torch.from_numpy(test_feats).float(), os.path.join(data_dir, 'test_feats'))
torch.save(torch.from_numpy(train_field).float(), os.path.join(data_dir, 'train_field'))
torch.save(torch.from_numpy(valid_field).float(), os.path.join(data_dir, 'valid_field'))
torch.save(torch.from_numpy(test_field).float(), os.path.join(data_dir, 'test_field'))
