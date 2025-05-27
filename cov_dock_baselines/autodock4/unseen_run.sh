time=$(date +'%Y%m%d_%H%M%S')
run_name="unseen_${time}"
out_dir="./res/${run_name}/"

# 1) prepare input
mkdir -p res
cp -r ../../data/processed ./res
mv ./res/processed ./res/$run_name
python prepare_docking_data.py --work-dir $out_dir --only-unseen 1
# failed pdb 3
# ['6ZBM', '7JS7', '7AWE']

# 2) run autodock4cov predict
python docking.py --work-dir $out_dir --only-unseen 1
# failed pdb 15
# ['7DT2', '7JN7', '7D9W', '7KRZ', '6WYH', '6ZBM', '7JS7', '6LX1', '7AWE', '7CC2', '6ZBW', '6WZV', '6X1M', '6X6C', '6ZBX']
