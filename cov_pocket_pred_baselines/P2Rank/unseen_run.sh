
seed=2
threads=10
time=$(date +'%Y%m%d_%H%M%S')
run_name="unseen_seed${seed}_threads${threads}_${time}"
out_dir="./res/${run_name}/"

# 1) prepare input for p2rank
python my_prepare_input.py --res-dir $out_dir --dataset_filename ../../data/processed/dataset.filtered.unseen.csv --only-unseen 1

# 2) run p2rank predict
seed=$seed threads=$threads out_dir=$out_dir bash run_p2rank.sh
