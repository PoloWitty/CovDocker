# eval
export WANDB_API_KEY=db0c63baeacf1eeb82545a162529728192b83020

# run_name="reactionT5_seed0_20240801_204309"
# run_name="reactionT5_seed1_20240801_210631"
run_name="reactionT5_seed2_20240801_222741"
model_path="./save/${run_name}/best_model"
infer_filename="${model_path}/infer_result/forward_reaction_prediction_output.csv"
test_filename='./data/input4ReactionT5.unseen.csv'

time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="unseen_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID

seed=2

WANDB_MODE='disabled' python prediction.py \
    --model_name_or_path $model_path \
    --input_data=$test_filename \
    --num_beams=10 \
    --num_return_sequences=10 \
    --batch_size=4 \
    --output_dir="${model_path}/infer_result/" \
    --seed $seed



python my_compute_res.py \
    --infer_file $infer_filename\
    --ref_file $test_filename \
    --use-wandb 1 \
    --run-id $run_name \
    --infer-config-filename "${model_path}/infer_result/pred_args.json"