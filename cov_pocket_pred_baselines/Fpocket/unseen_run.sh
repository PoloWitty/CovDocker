
# there is no seed parameter for fpocket
time=$(date +'%Y%m%d_%H%M%S')
run_name="unseen_${time}"
out_dir="./res/${run_name}/"

# 1) prepare input for fpocket
python my_prepare_input.py --res-dir $out_dir --dataset_filename ../../data/processed/dataset.filtered.unseen.csv --only-unseen 1

# 2) run fpocket predict
out_dir=$out_dir bash run_fpocket.sh

