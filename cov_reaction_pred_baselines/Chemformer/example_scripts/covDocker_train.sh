#!/bin/bash
# continue fine-tuning
# export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=db0c63baeacf1eeb82545a162529728192b83020
export WANDB_PROJECT=rebuttal_reaction_prediction
time=$(date +'%Y%m%d_%H%M%S')
[ -z "${seed}" ] && seed=0
export WANDB_RUN_ID="chemformer_seed${seed}_${time}"
export WANDB_NAME=$WANDB_RUN_ID
python -m molbart.fine_tune \
  --dataset_type covdocker_synthesis \
  --data_path ../../data/processed/dataset.csv \
  --model_path ./uspto_sep_finetuned_last.ckpt \
  --task forward_prediction \
  --vocabulary_path bart_vocab_downstream.json \
  --n_gpus 1 \
  --n_epochs 100 \
  --check_val_every_n_epoch 2 \
  --limit_val_batches 4 \
  -lr 0.001 \
  --schedule cycle \
  --batch_size 64 \
  --acc_batches 8 \
  --augmentation_strategy all \
  -aug_prob 0.5 \
  --model_type bart \
  --seed $seed \
  --use_wandb
  # --warm_up_steps 100 \
  # --weight_decay 2e-5 \
  # --dropout 0.14 \

# # train from scratch
# python -m molbart.fine_tune \
#   --dataset_type covdocker_synthesis \
#   --data_path ../data/processed/examples.pocket_with_smiles.random_split.aug_train.csv \
#   --task forward_prediction \
#   --vocabulary_path bart_vocab_covDocker.json \
#   --n_gpus 1 \
#   --n_epochs 250 \
#   --check_val_every_n_epoch 10 \
#   --limit_val_batches 4 \
#   -lr 0.001 \
#   --schedule cycle \
#   --batch_size 64 \
#   --acc_batches 8 \
#   --augmentation_strategy all \
#   -aug_prob 0.5 \
#   --model_type bart \
#   --warm_up_steps 600 \
#   --n_layers 6 \
#   --dropout 0.1 \
#   --use_wandb

