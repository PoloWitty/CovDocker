# run_name="seed0_20240802_052029"
# run_name="seed1_20240802_065321"
run_name="seed2_20240802_070137"
out_dir="./res/${run_name}/"


time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="vina_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID
# 3) gather result
python gather_res.py --work-dir $out_dir --run-id $run_name --use-wandb 1 --dock-log-filename "${out_dir}log.json"