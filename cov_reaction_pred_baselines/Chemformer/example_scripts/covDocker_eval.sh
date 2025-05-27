#!/bin/bash

run_name=chemformer_seed0
# run_name=chemformer_seed1
# run_name=chemformer_seed2
weight_path="../../covDocker_models/reaction_prediction/${run_name}/checkpoints/last.ckpt"

time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID

seed=0

# eval for continue fine-tuning
python -m molbart.evaluate \
  --dataset_type covdocker_synthesis \
  --data_path ../../data/processed/dataset.csv \
  --model_path  $weight_path\
  --task forward_prediction \
  --vocabulary_path bart_vocab_downstream.json \
  --n_gpus 1 \
  --batch_size 512 \
  --model_type bart \
  --n_beams 10 \
  --seed $seed \
  --use-wandb 0\
  --run-id $run_name