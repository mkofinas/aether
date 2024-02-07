#!/bin/bash

GPU=0 # Set to whatever GPU you want to use

# Make sure to replace this with the directory containing the data files
DATA_PATH='experiments/electrostatic/dataset/data/electrostatic_field/'

BASE_RESULTS_DIR="experiments/electrostatic/results"

SEED=1
MODEL_TYPE="nn.seq2seq.locs.LoCS"
EXPERIMENT_EXT=""
WORKING_DIR="${BASE_RESULTS_DIR}/${MODEL_TYPE}${EXPERIMENT_EXT}/seed_${SEED}/"
ENCODER_ARGS="--encoder_hidden 512 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 256 --encoder_rnn_hidden 128"
DECODER_ARGS="--decoder_hidden 512 --decoder_type recurrent"
HIDDEN_ARGS="--rnn_hidden 128"
PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 256"
MODEL_ARGS="--model_type ${MODEL_TYPE} --graph_type dynamic --num_edge_types 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
TRAINING_ARGS='--vel_norm_norm --add_uniform_prior --no_edge_prior 0.5 --batch_size 64 --lr 5e-4 --use_adam --num_epochs 600 --lr_decay_factor 0.5 --lr_decay_steps 600 --normalize_kl --normalize_nll --tune_on_mse --val_teacher_forcing --teacher_forcing_steps -1'
mkdir -p $WORKING_DIR
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/electrostatic/main.py --gpu \
  --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/electrostatic/main.py --gpu \
  --mode eval --report_error_norm --load_best_model --test_pred_steps 20 \
  --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/electrostatic/main.py --gpu \
  --mode save_pred --report_error_norm --load_best_model --test_pred_steps 20 \
  --data_path $DATA_PATH --working_dir $WORKING_DIR --end_idx 100 $MODEL_ARGS $TRAINING_ARGS
