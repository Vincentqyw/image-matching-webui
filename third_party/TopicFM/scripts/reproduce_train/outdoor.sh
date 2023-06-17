#!/bin/bash -l

SCRIPTPATH=$(dirname $(readlink -f "$0"))
PROJECT_DIR="${SCRIPTPATH}/../../"

# conda activate loftr
export PYTHONPATH=$PROJECT_DIR:$PYTHONPATH
cd $PROJECT_DIR

data_cfg_path="configs/data/megadepth_trainval.py"
main_cfg_path="configs/model/outdoor/model_ds.py"

n_nodes=1
n_gpus_per_node=4
torch_num_workers=4
batch_size=1
pin_memory=true
exp_name="outdoor-bs=$(($n_gpus_per_node * $n_nodes * $batch_size))"

python -u ./train.py \
    ${data_cfg_path} \
    ${main_cfg_path} \
    --exp_name=${exp_name} \
    --gpus=${n_gpus_per_node} --num_nodes=${n_nodes} --accelerator="ddp" \
    --batch_size=${batch_size} --num_workers=${torch_num_workers} --pin_memory=${pin_memory} \
    --check_val_every_n_epoch=1 \
    --log_every_n_steps=30000 \
    --flush_logs_every_n_steps=30000 \
    --limit_val_batches=1. \
    --num_sanity_val_steps=10 \
    --benchmark=True \
    --max_epochs=40 # --ckpt_path="pretrained_epoch22.ckpt"
