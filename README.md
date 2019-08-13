# GAN_mapping_relationship

This is the implementation of [our paper](https://arxiv.org/abs/1804.00316). 
In this paper, we proposed an unsupervised phoneme recogntion system which can achieve 36% phoneme accuracy on TIMIT with oracle phone boundaries.
This method developed a GAN-based model to achieve unsupervised phoneme recognition.

## How to use

### Dependencies
1. tensorflow 1.13

2. kaldi

3. librosa

### Data preprocess
- Usage:

1. Modify `path.sh` with your path of Kaldi and srilm.
2. Modify `config.sh` with your feature path and timit path.
3. Run `$ bash preprocess.sh`

- Phoneme sequences can download from [here](https://www.dropbox.com/s/rux7tnr0n6k6n33/phn_seq.tar.gz?dl=0), and put `fake.39` and `oracle.39` in `./data`.

### Train model
- Usage:

1. Modify the experimental and path setting in `config.sh`.
2. Modify the model's parameter in `src/audio2vec.sh` and `src/mapping.sh`.
2. Run `$ bash run.sh`

- This scipt contains the training flow of whole system.

## Hyperparameters in `config.sh`
`cluster_num` : number of cluster.

`target_type` : type of phoneme sequences (oracle/fake).

## Hyperparameters in `src/audio2vec.sh`
`mode` : train or test mode (train/test), test mode is the step of clustering.

`lr` : learning rate.

`max_length` : max length of acoustic token.

`hidden_units` : hidden size of the audio2vec.

`batch_size` : batch size.

`epoch` : number of training epoch.

`kl_saturate` : type of phoneme sequences (oracle/fake).

`kl_step` : type of phoneme sequences (oracle/fake).

`cuda_id` : GPU ids.

## Hyperparameters in `src/mapping.sh`
`mode` : train or test mode (train/test).

`generator_lr` : learning rate of generator.

`discriminator_lr` : learning rate of discriminator.

`max_length` : max length of phoneme sequence.

`step` : number of training step.

`discriminator_hidden_units` : hidden size of the discriminator.

`discriminator_iterations` : training iteration of discriminator.

`batch_size` : batch size.

`cuda_id` : GPU ids.

## Reference
[Completely Unsupervised Phoneme Recognition by Adversarially Learning Mapping Relationships from Audio Embeddings](https://arxiv.org/abs/1804.00316),  Da-Rong Liu, Kuan-Yu Chen *et.al.*

## Acknowledgement
**Special thanks to Da-Rong Liu !**



