export LOCAL_RANK=0

# infer on test set
data_path="./data/processed/dataset/reactive_site/"  # replace to your data path
run_name="ours_seed0"
results_path="./covDocker_models/reactive_site/infer"  # replace to your results path

###################### change this to other seed to test more result ######################
weight_path="./covDocker_models/reactive_site/checkpoint_seed0.pt"
##########################################################################################

batch_size=1
weighted_center=2
cross_attn_layers=1
seed=0

CUDA_VISIBLE_DEVICES=0 python ./unimol/infer.py --user-dir ./unimol $data_path --valid-subset test \
       --results-path $results_path \
       --num-workers 0 --ddp-backend=c10d --batch-size $batch_size \
       --task covDocker_reactive_site --loss token_clf_cross_entropy_pocket --arch covDocker_reactive_site_model_large \
       --path $weight_path \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 512 \
       --cross-attention-layers $cross_attn_layers \
       --weighted-center $weighted_center \
       --log-interval 50 --log-format simple \
       --seed $seed


time=$(date +'%Y%m%d_%H%M%S')
export WANDB_RUN_ID="${run_name}_eval${time}"
export WANDB_NAME=$WANDB_RUN_ID
# calc acc
predict_file="${results_path}/${run_name}_test.out.pkl"  # Your inference file dir

python ./unimol/utils/position_prediction_metrics.py --predict-file $predict_file --run-id $run_name --use-wandb 1 --infer-config-filename "${results_path}/infer_args.json"
