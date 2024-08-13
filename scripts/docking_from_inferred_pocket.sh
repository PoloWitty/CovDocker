export LOCAL_RANK=0

reactive_site_run_name="ours_seed0_20240731_235842"
# reactive_site_run_name="ours_seed1_20240801_030233"
# reactive_site_run_name="ours_seed2_20240801_045626"
results_path="./save/reactive_site/${reactive_site_run_name}/infer"  # replace to your results path
lmdb_path="${results_path}/inferred_pocket"
mkdir -p $lmdb_path

# prepare input from inferred pocket
python utils/infer_pocket_pdb.py --infer_res_pkl "${results_path}/${reactive_site_run_name}_test.out.pkl" --dataset_info_file ./data/processed/dataset.filtered.csv --save_dir $lmdb_path



# infer on test set
docking_run_name="ours_seed0_20240802_025357"
# docking_run_name="ours_seed1_20240802_031340"
# docking_run_name="ours_seed2_20240802_081642"
data_path="${lmdb_path}/"  # replace to your data path
cp ./data/processed/dataset/docking/dict_mol.txt $data_path
cp ./data/processed/dataset/docking/dict_pkt.txt $data_path
weight_path="./save/dock/${docking_run_name}/checkpoint_last.pt"
results_path="./save/dock/${docking_run_name}/pipe_infer"  # replace to your results path

batch_size=1
dist_threshold=8.0
recycling=3
min_dist_norm_weight=1.0
min_dist_norm_type=1
refine=1

seed=0

CUDA_VISIBLE_DEVICES=0 python ./unimol/infer.py --user-dir ./unimol $data_path --valid-subset test \
       --results-path $results_path \
       --num-workers 8 --ddp-backend=c10d --batch-size $batch_size \
       --task docking_pose_cov --loss docking_pose_cov --arch docking_pose_cov \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --dist-threshold $dist_threshold --recycling $recycling \
       --min-dist-norm-weight $min_dist_norm_weight \
       --min-dist-norm-type $min_dist_norm_type \
       --refine $refine \
       --log-interval 50 --log-format simple \
       --seed $seed



# calc rmsd

time=$(date +'%Y%m%d_%H%M%S')
run_name="site_${reactive_site_run_name}_dock_${docking_run_name}"
export WANDB_RUN_ID="pipe_post10sigma_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID


nthreads=20  # Num of threads
predict_file="${results_path}/${docking_run_name}_test.out.pkl"  # Your inference file dir
reference_file="${data_path}test.lmdb"  # Your reference file dir
output_path="${results_path}/protein_ligand_binding_pose_prediction"  # Docking results path

# rm -r $output_path
# mkdir -p $output_path

# echo $predict_file
# echo $reference_file
python ./unimol/utils/docking.py \
    --nthreads $nthreads --predict-file $predict_file --reference-file $reference_file --output-path $output_path --do-post-process 1 \
    --run-id $run_name --use-wandb 1 --infer-config-filename "${results_path}/infer_args.json" \
    --infer-time-filename "${results_path}/infer_time.json" \
    --pipe-dock 1

