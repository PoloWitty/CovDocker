
# eval


run_name="T5Chem_seed2_20240802_004959"
model_path="./save/${run_name}"

time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="unseen_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID

seed=2

# unseen.target or unseen.source is both ok, see ./t5chem/t5chem/run_prediction.py L102 for more detail
mkdir -p "${model_path}/infer_result"
python t5chem/t5chem/__main__.py predict \
    --data_dir data/covDocker/unseen.target \
    --model_dir $model_path \
    --prediction "${model_path}/infer_result/sample_res.csv" \
    --num_beams 10 \
    --num_preds 10 \
    --batch_size 12 \
    --use_wandb 1 \
    --train-config-filename "${model_path}/training_args.bin" \
    --seed $seed \
    --run-id $run_name