
[ -z "${seed}" ] && seed=2
time=$(date +'%Y%m%d_%H%M%S')
run_name="unseen_seed${seed}_${time}"

# 1) prepare input
mkdir -p res
cp -r ../../data/processed ./res
mv ./res/processed ./res/$run_name

# 2) run vina predict
python docking.py --seed $seed --work-dir "./res/$run_name" --only-unseen 1

# failed pdb 8
# ['7DT2', '7JN7', '7KRZ', '7AWE', '7CC2', '6WZV', '6X1M', '6X6C']
# total run time: 749s
# average run time: 12.48s