

run_name="unseen_seed2_threads10_20240805_000046"
out_dir="./res/${run_name}/"

# config used when run p2rank
seed=2
threads=10

time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="p2rank_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID
# 3) gather result
python my_gather_res.py --res-dir $out_dir --run-id $run_name --use-wandb 1 \
 --run-seed $seed --run-thread $threads --only-unseen 1 \
 --dataset_filename ../../data/processed/dataset.filtered.unseen.csv