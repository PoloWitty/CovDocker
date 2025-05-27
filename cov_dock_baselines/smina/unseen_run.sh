
[ -z "${seed}" ] && seed=2
time=$(date +'%Y%m%d_%H%M%S')
run_name="unseen_seed${seed}_${time}"

# 1) prepare input
mkdir -p res
cp -r ../../data/processed ./res
mv ./res/processed ./res/$run_name

# 2) run smina predict
python docking.py --seed $seed --work-dir "./res/$run_name" --only-unseen 1

# failed pdb 0
# []
# mean time cost 15.346009135246277s
# total time cost 1043.5286211967468s