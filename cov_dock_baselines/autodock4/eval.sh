export WANDB_API_KEY=db0c63baeacf1eeb82545a162529728192b83020

# run_name="20240802_181430"
# run_name="20240802_200401"
run_name="20240802_205753"
out_dir="./res/${run_name}/"


time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="autodock4cov_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID
# 3) gather result
python gather_res.py --work-dir $out_dir --run-id $run_name --use-wandb 1