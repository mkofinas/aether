### Lorentz force field

The codebase for the Lorentz force field experiment is based on [SE(3) Equivariant Graph
Neural Networks with Complete Local Frames](https://arxiv.org/abs/2110.14811) (**ClofNet**),
by Weitao Du, He Zhang, Yuanqi Du, Qi Meng, Wei Chen, Bin Shao, Tie-Yan Liu, ICML 2022.

## Run the code

### Newtonian many-body system

This task is inspired by Kipf et al., 2018; Fuchs et al., 2020; Satorras et al., 2021b,
where a 5-body charged system is controlled by the electrostatic force field. The
original source code for generating trajectories comes from
[NRI](https://github.com/ethanfetaya/NRI) and is modified by
[EGNN](https://github.com/vgsatorras/egnn). [ClofNet](https://arxiv.org/abs/2110.14811)
further extends the version of EGNN to three new settings, imposing external force
fields into the original system, a gravity field and a Lorentz-like dynamical force
field, which can provide more complex and dynamical force directions.

We sincerely thank the solid contribution of these 3 works.

#### Create Lorentz dataset
```
cd dataset
bash script.sh
```

#### Run experiment
```
python -u main.py --max_training_samples 3000 --norm_diff True --LR_decay True --decay 0.9 --lr 0.005 --outf saved \
  --data_mode dynamic_20body --epochs 600 --exp_name aether_dynamic_20body --model aether --n_layers 4 --data_root dataset/data
```
Change the `data_root` variable if you store the dataset in a different directory.
