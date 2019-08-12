#!/bin/bash

# Data path and overall setting
. ./config.sh

mode='test' 

# audio2vec model setting
max_length=20
hidden_units=512
batch_size=128
cuda_id=0

# model saving path
save_dir="${save_path}/audio2vec"
cluster_dir="${save_path}/cluster"


cd audio2vec
CUDA_VISIBLE_DEVICES="$cuda_id" python3 main.py --mode $mode --output \
--max_length $max_length --hidden_units $hidden_units \
--batch_size $batch_size --save_dir $save_dir --cluster_dir $cluster_dir \
--train_feat $train_feat --test_feat $test_feat --train_phn $train_phn --test_phn $test_phn \
--meta $meta --cluster_num $cluster_num

cd ..