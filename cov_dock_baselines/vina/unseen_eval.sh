
run_name="unseen_seed2_20240805_015737"
out_dir="./res/${run_name}/"


time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="vina_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID
# 3) gather result
python gather_res.py --work-dir $out_dir --run-id $run_name --use-wandb 1 --dock-log-filename "${out_dir}log.json" --only-unseen 1