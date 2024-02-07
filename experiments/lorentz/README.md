## Lorentz force field

The codebase for the Lorentz force field experiment is based on [SE(3) Equivariant Graph
Neural Networks with Complete Local Frames](https://arxiv.org/abs/2110.14811) (**ClofNet**),
by Weitao Du, He Zhang, Yuanqi Du, Qi Meng, Wei Chen, Bin Shao, Tie-Yan Liu, ICML 2022.

This task is inspired by [NRI](https://github.com/ethanfetaya/NRI),
where a 5-body system of charged particles evolves via the electrostatic forces between
particles. The original source code is modified by
[EGNN](https://github.com/vgsatorras/egnn). [ClofNet](https://github.com/mouthful/ClofNet)
further extends the version of EGNN to new settings, imposing an external Lorentz force
field into the original system, which provides more complex and dynamical force directions.

We sincerely thank the solid contribution of these 3 works.

#### Create Lorentz dataset
To create the Lorentz dataset, simply run the following:
```sh
cd dataset
bash script.sh
```

#### Run experiment
The following command will train a 4-layer Aether model on the Lorentz dataset:
```python
python -u main.py --max_training_samples 3000 --norm_diff True --LR_decay True \
  --decay 0.9 --lr 0.005 --outf saved --data_mode dynamic_20body --epochs 600 \
  --exp_name aether_dynamic_20body --model aether --n_layers 4 --data_root dataset/data
```
Change the `data_root` variable if you store the dataset in a different directory.

You can also change the `exp_name` and the `model` variable to `locs`, `egnn`, `clof_vel` to run the
corresponding model. The following command will train ClofNet on the Lorentz dataset:

```python
python -u main.py --max_training_samples 3000 --norm_diff True --LR_decay True \
  --decay 0.9 --lr 0.01 --outf saved --data_mode dynamic_20body --epochs 600 \
  --exp_name clof_vel_dynamic_20body --model clof_vel --n_layers 4 --data_root dataset/data
```
