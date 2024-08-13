# pip install sacrebleu
# pip install sentencepiece
# pip install accelerate -U
# pip install datasets

export WANDB_API_KEY=db0c63baeacf1eeb82545a162529728192b83020
export WANDB_PROJECT=rebuttal_reaction_prediction
export TOKENIZERS_PARALLELISM=true

time=$(date +'%Y%m%d_%H%M%S')
[ -z "${seed}" ] && seed=2
run_name="reactionT5_seed${seed}_${time}"
export WANDB_RUN_ID=$run_name
export WANDB_NAME=$WANDB_RUN_ID

train_filename='./data/input4ReactionT5.train.csv'
test_filename='./data/input4ReactionT5.test.csv'

# train
python finetune-pretrained-ReactionT5.py \
    --model=$run_name \
    --epochs=50 \
    --batch_size=16 \
    --train_data_path=$train_filename \
    --valid_data_path=$test_filename \
    --fp16 \
    --seed $seed


