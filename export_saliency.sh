#!/bin/bash

hidden_dim=$1
p=$2
arch=ResRNANet
ss_type=seq

name=$p"_"$1
exp=$name

CUDA_VISIBLE_DEVICES=1 python -u export_saliency.py \
    --arch $arch \
    --ss_type $ss_type \
    --out_dir models/$exp \
    --exp_name $exp\
    --hidden_dim $hidden_dim \
    --p_name $p\
    --restore_best \
