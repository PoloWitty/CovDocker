data_path="./pocket_property_prediction"  # replace to your data path
save_dir="./save_finetune"  # replace to your save path
n_gpu=1
MASTER_PORT=10086
dict_name="dict_coarse.txt"
weight_path="./weights/checkpoint.pt"
task_name="druggability"  # or "nrdld", pocket property prediction dataset folder name 
lr=3e-4
batch_size=32
epoch=20
dropout=0
warmup=0.1
local_batch_size=32
seed=1

if [ "$task_name" == "druggability" ]; then
       metric="valid_rmse"
       loss_func="finetune_mse_pocket"
       task_num=1
       fpocket_score="Druggability Score"  # choose in ["Score", "Druggability Score", "Total SASA", "Hydrophobicity score"]
else
       metric="loss"
       loss_func="finetune_cross_entropy_pocket"
       task_num=2
fi

export NCCL_ASYNC_ERROR_HANDLING=1
export OMP_NUM_THREADS=1
update_freq=`expr $batch_size / $local_batch_size`
python -m torch.distributed.launch --nproc_per_node=$n_gpu --master_port=$MASTER_PORT $(which unicore-train) $data_path --task-name $task_name --user-dir ./unimol --train-subset train --valid-subset valid \
       --num-workers 8 --ddp-backend=c10d \
       --dict-name $dict_name \
       --task pocket_finetune --loss $loss_func --arch unimol_base  \
       --classification-head-name $task_name --num-classes $task_num \
       --optimizer adam --adam-betas "(0.9, 0.99)" --adam-eps 1e-6 --clip-norm 1.0 \
       --lr-scheduler polynomial_decay --lr $lr --warmup-ratio $warmup --max-epoch $epoch --batch-size $local_batch_size --pooler-dropout $dropout \
       --update-freq $update_freq --seed $seed \
       --fp16 --fp16-init-scale 4 --fp16-scale-window 256 \
       --log-interval 100 --log-format simple \
       --validate-interval 1 --finetune-from-model $weight_path \
       --best-checkpoint-metric $metric --patience 2000 \
       --save-dir $save_dir --remove-hydrogen --fpocket-score "$fpocket_score"

# --maximize-best-checkpoint-metric, for classification task
