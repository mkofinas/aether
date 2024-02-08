import os
import pickle
import torch
import numpy as np

data_dir = 'dataset/data/electrostatic_field'
simulation = 'electrostatic_field'
ndim = 2
particles = 5
with open(os.path.join(data_dir, f'ds_train_{simulation}_{ndim}D_{particles}_new.pkl'), 'rb') as f:
    train_data = pickle.load(f)

with open(os.path.join(data_dir, f'ds_valid_{simulation}_{ndim}D_{particles}_new.pkl'), 'rb') as f:
    valid_data = pickle.load(f)

with open(os.path.join(data_dir, f'ds_test_{simulation}_{ndim}D_{particles}_new.pkl'), 'rb') as f:
    test_data = pickle.load(f)

train_feats = np.concatenate([train_data['points'], train_data['vel']], -2)
valid_feats = np.concatenate([valid_data['points'], valid_data['vel']], -2)
test_feats = np.concatenate([test_data['points'], test_data['vel']], -2)

static_field = train_feats[[0], [0], particles:, :]
train_feats = train_feats[..., :particles, :]
valid_feats = valid_feats[..., :particles, :]
test_feats = test_feats[..., :particles, :]

torch.save(torch.from_numpy(train_feats).float(), os.path.join(data_dir, 'train_feats'))
torch.save(torch.from_numpy(valid_feats).float(), os.path.join(data_dir, 'valid_feats'))
torch.save(torch.from_numpy(test_feats).float(), os.path.join(data_dir, 'test_feats'))
torch.save(torch.from_numpy(static_field).float(), os.path.join(data_dir, 'static_field'))

train_edges = train_data['edges']
valid_edges = valid_data['edges']
test_edges = test_data['edges']

torch.save(torch.from_numpy(train_edges).float(), os.path.join(data_dir, 'train_edges'))
torch.save(torch.from_numpy(valid_edges).float(), os.path.join(data_dir, 'valid_edges'))
torch.save(torch.from_numpy(test_edges).float(), os.path.join(data_dir, 'test_edges'))

train_charges = np.stack(train_data['charges'])
valid_charges = np.stack(valid_data['charges'])
test_charges = np.stack(test_data['charges'])
static_charges = train_charges[0, particles:, :]

torch.save(torch.from_numpy(train_charges).squeeze().float(), os.path.join(data_dir, 'train_charges'))
torch.save(torch.from_numpy(valid_charges).squeeze().float(), os.path.join(data_dir, 'valid_charges'))
torch.save(torch.from_numpy(test_charges).squeeze().float(), os.path.join(data_dir, 'test_charges'))
torch.save(torch.from_numpy(static_charges).float(), os.path.join(data_dir, 'static_charges'))
