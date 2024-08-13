#!/bin/bash
export WANDB_API_KEY=db0c63baeacf1eeb82545a162529728192b83020



# prepare unseen data for chemformer
unseen_filename='./unseen_data.csv'
python prepare_unseen_data.py --dataset_info_file ../../data/processed/dataset.unseen.csv --save_filename $unseen_filename


# run_name=chemformer_seed0_20240801_093759
# run_name=chemformer_seed1_20240801_100205
run_name=chemformer_seed2_20240801_102626
weight_path="./outputs/rebuttal_reaction_prediction/${run_name}/checkpoints/last.ckpt"

time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="unseen_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID

seed=2

# eval for continue fine-tuning
python -m molbart.evaluate \
  --dataset_type covdocker_synthesis \
  --data_path $unseen_filename \
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