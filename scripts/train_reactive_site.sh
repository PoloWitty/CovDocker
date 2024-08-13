[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${MASTER_PORT}" ] && MASTER_PORT=10087

# n_gpu=1 # for debug
unset WANDB_NAME
unset WANDB_JOB_NAME
unset WANDB_RUN_GROUP
unset WANDB_RUN_ID
data_path="./data/processed/dataset/reactive_site/"  # replace to your data path
task_name="covDocker_reactive_site"  # or "nrdld", pocket property prediction dataset folder name 
finetune_mol_model="./unimol_weight/mol_pre_no_h_220816.pt"
lr=1e-4
wd=1e-4
batch_size=4
local_batch_size=1
epoch=150
dropout=0.3
warmup=0.1
seed=2

# pocket_token_clf_loss_weight is not used in the final version (see unimol/losses/cross_entropy.py L379)
[ -z "${pocket_token_clf_loss_weight}" ] && pocket_token_clf_loss_weight=0.5 

[ -z "${pocket_coord_loss_weight}" ] && pocket_coord_loss_weight=1.0
[ -z "${reactive_loss_weight}" ] && reactive_loss_weight=0.05
[ -z "${weighted_center}" ] && weighted_center=2
[ -z "${cross_attn_layers}" ] && cross_attn_layers=1


time=$(date +'%Y%m%d_%H%M%S')
run_name="ours_seed${seed}_${time}"
save_dir="./save/reactive_site/${run_name}"
rm -rf ${save_dir} # run from scratch
mkdir -p ${save_dir}


metric="infer_reactive_acc"
loss_func="token_clf_cross_entropy_pocket"
task_num=1

export NCCL_IB_DISABLE=1
export NCCL_SOCKET_IFNAME=eth0
export NCCL_P2P_DISABLE=1
export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 10 --ddp-backend=c10d \
       --task $task_name --loss $loss_func --arch covDocker_reactive_site_model_large \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 --weight-decay $wd \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 10 --log-format simple \
       --save-interval 10 --validate-interval 1 \
       --best-checkpoint-metric $metric --patience 2000 \
       --save-dir $save_dir --remove-hydrogen \
       --classification-head-name $task_name --num-classes $task_num \
       --tensorboard-logdir $save_dir/tsb \
       --finetune-mol-model $finetune_mol_model \
       --pocket-token-clf-loss-weight $pocket_token_clf_loss_weight --pocket-coord-loss-weight $pocket_coord_loss_weight --reactive-loss-weight $reactive_loss_weight\
       --pooler-dropout $dropout \
       --weighted-center $weighted_center \
       --cross-attention-layers $cross_attn_layers \
       --all-gather-list-size 1000000 \
