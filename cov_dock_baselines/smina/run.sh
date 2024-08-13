
[ -z "${seed}" ] && seed=1
time=$(date +'%Y%m%d_%H%M%S')
run_name="seed${seed}_${time}"

# 1) prepare input
mkdir -p res
cp -r ../../data/processed ./res
mv ./res/processed ./res/$run_name

# 2) run smina predict
python docking.py --seed $seed --work-dir "./res/$run_name"