# !/bin/bash

# EXP SETTING
export cluster_num=300
export target_type=oracle #oracle/fake

# PATH
export root_path=/home/guanyu/guanyu/interspeech2018/handoff
export data_path=${root_path}/data
export save_path=${root_path}/save/${target_type}_${cluster_num}

# TIMIT 
export timit_path=/home/guanyu/guanyu/timit/timit_feature
export train_feat=${timit_path}/timit-train-mfcc-nor.pkl
export train_phn=${timit_path}/timit-train-phn.pkl
export test_feat=${timit_path}/timit-test-mfcc-nor.pkl
export test_phn=${timit_path}/timit-test-phn.pkl
export meta=${timit_path}/timit-train-meta.pkl
