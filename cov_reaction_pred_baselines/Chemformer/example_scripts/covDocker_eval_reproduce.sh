#!/bin/bash
export WANDB_API_KEY=db0c63baeacf1eeb82545a162529728192b83020

run_name=chemformer_seed0
###################### change this to other seed to test more result ######################
weight_path="../../covDocker_models/chemformer/checkpoint_seed0.ckpt"
##########################################################################################

time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID

seed=2

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
  --use-wandb 1\
  --run-id $run_name