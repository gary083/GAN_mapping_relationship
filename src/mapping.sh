#!/bin/bash

# Data path and overall setting
. ./config.sh

mode='train' 

train_idx="${save_path}/cluster/train.cluster"
train_audio2vec="${save_path}/cluster/train.audio2vec"
train_oracle="${save_path}/cluster/train.phn"
test_idx="${save_path}/cluster/test.cluster"
test_audio2vec="${save_path}/cluster/test.audio2vec"
test_oracle="${save_path}/cluster/test.phn"

mapping="${save_path}/cluster/best.map"
target="${data_path}/${target_type}.39"


# matching network model setting
generator_lr=0.01
discriminator_lr=0.001
max_length=120
batch_size=128
step=10000
discriminator_hidden_units=256
discriminator_iterations=3
cuda_id=0

# model saving path
save_dir="${save_path}/matching_network"
result_file="${save_path}/result.txt"

cd matching_network
CUDA_VISIBLE_DEVICES="$cuda_id" python3 main.py --mode $mode \
--generator_lr $generator_lr --discriminator_lr $discriminator_lr \
--max_length $max_length --batch_size $batch_size --step $step \
--discriminator_hidden_units $discriminator_hidden_units \
--discriminator_iterations $discriminator_iterations \
--result_file $result_file --save_dir $save_dir \
--train_idx $train_idx --train_audio2vec $train_audio2vec --train_oracle $train_oracle \
--test_idx $test_idx --test_audio2vec $test_audio2vec --test_oracle $test_oracle \
--mapping $mapping --target $target 

cd ..