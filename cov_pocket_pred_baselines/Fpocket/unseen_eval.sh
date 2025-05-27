

run_name="unseen_20240805_000937"
out_dir="./res/${run_name}/"


time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="fpocket_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID
# 3) gather result
python my_gather_res.py --res-dir $out_dir --run-id $run_name --use-wandb 1 \
 --only-unseen 1 \
 --dataset_filename ../../data/processed/dataset.filtered.unseen.csv