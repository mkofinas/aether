#!/bin/bash

set -x

GPU=0 # Set to whatever GPU you want to use

# First: process data

# Make sure to replace this with the directory containing the data files
DATA_PATH='experiments/ind/dataset/data/single_ind/'

BASE_RESULTS_DIR="experiments/ind/results/single_ind_processed"

SEED=1
MODEL_TYPE="nn.dynamicvars.aether_dynamicvars.AetherDynamicVars"
EXPERIMENT_EXT=""
WORKING_DIR="${BASE_RESULTS_DIR}/${MODEL_TYPE}${EXPERIMENT_EXT}/seed_${SEED}/"
ENCODER_ARGS="--encoder_hidden 256 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 128 --encoder_rnn_hidden 64 --encoder_normalize_mode normalize_all --normalize_inputs"
DECODER_ARGS="--decoder_hidden 256"
HIDDEN_ARGS="--rnn_hidden 64 --field_hidden 512"
PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 128"
MODEL_ARGS="--model_type $MODEL_TYPE --graph_type dynamic --skip_first --num_edge_types 4 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"

EVAL_ARGS="--eval_sequentially --present_gnn --strict 6 --report_error_norm --vel_norm_norm --train_data_len 20 --batch_size 1 --max_burn_in_count 6 --verbose --final_test --error_out_name test_prediction_errors_burnin6.npy --load_best_model"
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/ind/visualize_predictions.py --gpu \
  --mode eval --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $EVAL_ARGS
