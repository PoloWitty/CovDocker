export LOCAL_RANK=0

# infer on test set
data_path="./data/processed/dataset/docking/"  # replace to your data path
run_name="ours_seed0"
results_path="./covDocker_models/docking_pose_cov/infer_pose"  # replace to your results path

###################### change this to other seed to test more result ######################
weight_path="./covDocker_models/docking_pose_cov/checkpoint_seed0.pt"
##########################################################################################

batch_size=1
dist_threshold=8.0
recycling=3
min_dist_norm_weight=1
min_dist_norm_type=1
refine=1
[ -z "${do_post_process}" ] && do_post_process=0

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
export WANDB_RUN_ID="post10sigma_${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID

nthreads=20  # Num of threads
predict_file="${results_path}/${run_name}_test.out.pkl"  # Your inference file dir
reference_file="${data_path}test.lmdb"  # Your reference file dir
output_path="${results_path}/protein_ligand_binding_pose_prediction"  # Docking results path

python ./unimol/utils/docking.py \
 --nthreads $nthreads --predict-file $predict_file --reference-file $reference_file --output-path $output_path --do-post-process $do_post_process \
 --run-id $run_name --use-wandb 1 --infer-config-filename "${results_path}/infer_args.json" \
 --infer-time-filename "${results_path}/infer_time.json"
