# check whether n_gpu exist in env, if not, set to the number of GPUs
[ -z "${n_gpu}" ] && n_gpu=$(nvidia-smi -L | wc -l)
[ -z "${MASTER_PORT}" ] && MASTER_PORT=10086

# n_gpu=1 # for debug
unset WANDB_NAME
unset WANDB_JOB_NAME
unset WANDB_RUN_GROUP
unset WANDB_RUN_ID
data_path="./data/processed/dataset/docking/"  # replace to your data path
task_name="docking_pose"  # or "nrdld", pocket property prediction dataset folder name 
[ -z "${LR}" ] && LR=3e-4
finetune_mol_model="./unimol_weight/mol_pre_no_h_220816.pt"
finetune_pocket_model="./unimol_weight/pocket_pre_220816.pt"
batch_size=4
local_batch_size=4
epoch=50
dropout=0.2
warmup=0.06
[ -z "${seed}" ] && seed=0
dist_threshold=8.0
recycling=3

time=$(date +'%Y%m%d_%H%M%S')
run_name="unimol_seed${seed}_${time}"
save_dir="./save/dock/${run_name}"
rm -rf ${save_dir} # run from scratch
mkdir -p ${save_dir}

metric="valid_loss"
loss_func="docking_pose"

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
torchrun --nproc_per_node=$n_gpu --standalone --nnodes=1 $(which unicore-train) $data_path --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --task $task_name --loss $loss_func --arch docking_pose \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $LR --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size  \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 10 --log-format simple \
       --save-interval 10 --validate-interval 1 \
       --best-checkpoint-metric $metric --patience 2000 \
       --save-dir $save_dir \
       --finetune-mol-model $finetune_mol_model \
       --finetune-pocket-model $finetune_pocket_model \
       --dist-threshold $dist_threshold --recycling $recycling \
       --tensorboard-logdir $save_dir/tsb \
       --find-unused-parameters \
       --all-gather-list-size 10000000 \
       --mol-pooler-dropout $dropout --pocket-pooler-dropout $dropout \
       --wandb-project "polowitty/rebuttal_dock" --wandb-name $run_name\
