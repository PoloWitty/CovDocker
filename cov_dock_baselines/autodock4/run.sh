time=$(date +'%Y%m%d_%H%M%S')
run_name="${time}"
out_dir="./res/${run_name}/"

# 1) prepare input
mkdir -p res
cp -r ../../data/processed ./res
mv ./res/processed ./res/$run_name
python prepare_docking_data.py --work-dir $out_dir

# 2) run autodock4cov predict
python docking.py --work-dir $out_dir

