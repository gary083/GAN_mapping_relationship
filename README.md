# GAN_mapping_relationship

This is the implementation of [our paper](https://arxiv.org/abs/1804.00316). In this paper, we proposed an unsupervised phoneme recogntion system which can achieve 33.1% phoneme error rate on TIMIT.
This method developed a GAN-based model to achieve unsupervised phoneme recognition and we further use a set of HMMs to work in harmony with the GAN.

## How to use

### Dependencies
1. tensorflow 1.13

2. kaldi

3. srilm (can be built with kaldi/tools/install_srilm.sh)

4. librosa

### Data preprocess
- Usage:

1. Modify `path.sh` with your path of Kaldi and srilm.
2. Modify `config.sh` with your code path and timit path.
3. Run `$ bash preprocess.sh`

- This script will extract features and split dataset into train/test set.

### Train model
- Usage:

1. Modify the experimental and path setting in `config.sh`.
2. Modify the model's parameter in `src/audio2vec.sh` and `src/mapping.sh`.
2. Run `$ bash run.sh`

- This scipt contains the training flow of whole system.

## Hyperparameters in `config.sh`
`cluster_num` : type of initial phoneme boundaries (orc/uns).

`target_type` : type of initial phoneme boundaries (orc/uns).

## Reference
[Completely Unsupervised Phoneme Recognition by Adversarially Learning Mapping Relationships from Audio Embeddings](https://arxiv.org/abs/1804.00316),  Da-Rong Liu, Kuan-Yu Chen *et.al.*

## Acknowledgement
**Special thanks to Da-Rong Liu !**



