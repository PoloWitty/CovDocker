

run_name="20240801_192615"
out_dir="./res/${run_name}/"


time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="fpocket_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID
# 3) gather result
python my_gather_res.py --res-dir $out_dir --run-id $run_name --use-wandb 1