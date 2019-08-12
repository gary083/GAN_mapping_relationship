#!/bin/bash

# Data path and overall setting
. ./config.sh

mode='train' 

# audio2vec model setting
lr=0.001
max_length=20
hidden_units=512
batch_size=128
epoch=200
kl_saturate=3
kl_step=50000
cuda_id=0

# model saving path
log="${save_path}/audio2vec/log.txt"
save_dir="${save_path}/audio2vec"

cd audio2vec
CUDA_VISIBLE_DEVICES="$cuda_id" python3 main.py --mode $mode \
--lr $lr --max_length $max_length \
--hidden_units $hidden_units --batch_size $batch_size \
--epoch $epoch --log $log --save_dir $save_dir \
--kl_saturate $kl_saturate --kl_step $kl_step \
--train_feat $train_feat --test_feat $test_feat --train_phn $train_phn --test_phn $test_phn \
--meta $meta --cluster_num $cluster_num
cd ..