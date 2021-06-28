#!/bin/bash
echo `date +%Y%m%d%H%M%S`
n=1
hidden_dim=$1
p=$2
arch=ResRNANet
ss_type=seq
opt=adam
name=$p"_"$1
exp=$name


CUDA_VISIBLE_DEVICES=1 python -u train.py \
    --datadir data \
    --arch $arch \
    --ss_type $ss_type \
    --out_dir models/$exp \
    --optimizer $opt \
    --batch_size 32 \
    --ngpu $n \
    --hidden_dim $1 \
    --log_interval 100\
    --exp_name $exp\
    --p_name $p\
    --lr 0.0003 \
    --weight_decay 0.000001\
    --niter 800 \
    --use_label \
    --tfboard \
    | tee models/${exp}.log


# --restore_best \
# --generate_data \
