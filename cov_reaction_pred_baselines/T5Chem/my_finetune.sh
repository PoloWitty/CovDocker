

# train

export WANDB_API_KEY=db0c63baeacf1eeb82545a162529728192b83020
export WANDB_PROJECT=rebuttal_reaction_prediction
export TOKENIZERS_PARALLELISM=true

time=$(date +'%Y%m%d_%H%M%S')
seed=2
run_name="T5Chem_seed${seed}_${time}"
export WANDB_RUN_ID=$run_name
export WANDB_NAME=$WANDB_RUN_ID

python t5chem/t5chem/__main__.py train \
    --data_dir data/covDocker/ \
    --output_dir "save/${run_name}" \
    --task_type product \
    --pretrain ./models/USPTO_500_MT \
    --num_epoch 30 \
    --batch_size 48 \
    --random_seed $seed


