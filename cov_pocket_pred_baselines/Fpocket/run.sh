
time=$(date +'%Y%m%d_%H%M%S')
run_name="${time}"
out_dir="./res/${run_name}/"

# 1) prepare input for fpocket
python my_prepare_input.py --res-dir $out_dir

# 2) run fpocket predict
out_dir=$out_dir bash run_fpocket.sh

