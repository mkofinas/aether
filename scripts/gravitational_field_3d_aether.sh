#!/bin/bash

GPU=0 # Set to whatever GPU you want to use

# Make sure to replace this with the directory containing the data files
DATA_PATH='experiments/gravitational/dataset/data/gravitational_field_3d/'

BASE_RESULTS_DIR="experiments/gravitational/results/gravitational_field_3d"

SEED=1
MODEL_TYPE="nn.seq2seq.dynamic_field_aether.DynamicFieldAether"
EXPERIMENT_EXT=""
WORKING_DIR="${BASE_RESULTS_DIR}/${MODEL_TYPE}${EXPERIMENT_EXT}/seed_${SEED}/"
ENCODER_ARGS="--encoder_hidden 512 --encoder_mlp_num_layers 3 --encoder_mlp_hidden 256 --encoder_rnn_hidden 128"
DECODER_ARGS="--decoder_hidden 512 --decoder_type recurrent"
HIDDEN_ARGS="--rnn_hidden 128 --mlp_hidden 512 --graph_hidden 512"
PRIOR_ARGS="--use_learned_prior --prior_num_layers 3 --prior_hidden_size 256"
MODEL_ARGS="--model_type ${MODEL_TYPE} --graph_type dynamic --num_edge_types 2 $ENCODER_ARGS $DECODER_ARGS $HIDDEN_ARGS $PRIOR_ARGS --seed ${SEED}"
TRAINING_ARGS='--use_3d --normalization speed_norm --add_uniform_prior --no_edge_prior 0.5 --batch_size 32 --val_batch_size 32 --lr 5e-4 --use_adam --num_epochs 400 --lr_decay_factor 0.5 --lr_decay_steps 200 --normalize_kl --normalize_nll --tune_on_mse --val_teacher_forcing --teacher_forcing_steps -1'
mkdir -p $WORKING_DIR
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/gravitational/main.py --gpu \
  --mode train --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/gravitational/main.py  --gpu \
  --mode eval --report_error_norm --load_best_model --test_pred_steps 5 \
  --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/gravitational/main.py --gpu \
  --mode save_pred --report_error_norm --load_best_model --test_pred_steps 5 \
  --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS
CUDA_VISIBLE_DEVICES=$GPU python -u experiments/gravitational/main.py --gpu \
  --mode visualize_field --report_error_norm --load_best_model --test_pred_steps 5 \
  --eval_split test --data_path $DATA_PATH --working_dir $WORKING_DIR $MODEL_ARGS $TRAINING_ARGS
