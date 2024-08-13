
# export CUDA_VISIBLE_DEVICES=0
export WANDB_API_KEY=478b429a65e97bbfdaf3c9941af62d63abc05816
export WANDB_PROJECT=covDocker_confidence_model
unset WANDB_NAME
unset WANDB_RUN_ID

# build dataset for  confidence model
python -m molbart.build_confidence_dataset \
  --data_path ../data/processed/dataset.filtered.random_split.csv \
  --model_path Chemformer/outputs/covDocker_reaction_prediction/20240427/checkpoints/epoch=113-step=569.ckpt \
  --vocabulary_path bart_vocab_downstream.json \
  --n_gpus 8 \
  --model_type bart \
  --batch_size 32 \
  --n_beams 10 \
  --dataset_part full \
  --save_path ./tmp/confidence_data.csv

# for temp in {0.1,0.05,0.01}; do
#   export WANDB_NAME='20240412-temp'$temp
#   echo $WANDB_NAME
  # python -m molbart.train_confidence_model \
  #   --dataset_type covdocker_confidence \
  #   --data_path ./tmp/confidence_data_top10.csv \
  #   --task forward_prediction \
  #   --vocabulary_path bart_vocab_covDocker.json \
  #   --n_gpus 1 \
  #   --n_epochs 150 \
  #   --check_val_every_n_epoch 2 \
  #   --limit_val_batches 4 \
  #   -lr 0.001 \
  #   --schedule cycle \
  #   --batch_size 128 \
  #   --acc_batches 8 \
  #   --augmentation_strategy all \
  #   --model_type bert \
  #   --warm_up_steps 50 \
  #   --dropout 0.2 \
  #   --temperature $temp \
  #   --use_wandb 
  #   # --weight_decay 2e-5 \
  #   # --model_path ./outputs/covDocker_confidence_model/sgoaeilb/checkpoints/epoch=139-step=1679.ckpt \
# done

# python -m molbart.confidence_predict \
#   --data_path ./tmp/confidence_data.csv \
#   --dataset_type covdocker_confidence \
#   --dist_output_path ./tmp/confidence_dist_pred.csv \
#   --model_path ./outputs/covDocker_confidence_model/l6xgofxb/checkpoints/epoch=139-step=979.ckpt \
#   --batch_size 64

# # rank
# python -m molbart.re_rank \
#   --model_path ./outputs/covDocker_confidence_model/sgoaeilb/checkpoints/epoch=139-step=1679.ckpt \
#   --vocabulary_path bart_vocab_covDocker.json \
#   --sampled_test_path ./sampled_test.pkl
