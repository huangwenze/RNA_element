#!/bin/bash
echo `date +%Y%m%d%H%M%S`

p=$1
ss_type=seq
exp=$name


CUDA_VISIBLE_DEVICES=0 python -u train.py \
    --datadir data \
    --ss_type $ss_type \
    --p_name $p \
    --generate_data \
    --use_label \
