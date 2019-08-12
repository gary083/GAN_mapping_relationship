import os
import sys
import _pickle as cPickle
import numpy as np
import random

class data_loader:
    def __init__(self, max_length, idx_data_path, audio2vec_data_path, oracle_data_path, mapping_path, target_data_path=None):
        self.load_target_vocab()
        self.load_mapping(mapping_path)
        self.max_length = max_length

        print('load idx data')
        idx_data = self.load_pickle(idx_data_path) #list of list
        audio2vec_data = self.load_pickle(audio2vec_data_path) #list of numpy
        oracle_data = self.load_pickle(oracle_data_path) #list of phn
        self.process_source_data(idx_data, audio2vec_data, oracle_data)

        if target_data_path:
            print('load phn data')
            target_data = self.load_target_data(target_data_path)
            self.process_target_data(target_data)

        print("finish loading data")

    def load_target_vocab(self):
        """
        read (voacb count) file and generate self.word2idx, self.idx2word
        """
        vocab = [line.split()[0] for line in open(os.path.join('preprocessed', 'all_vocab.txt'), 'r').read().splitlines()]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(self.word2idx)
        print('vocab size: ', self.vocab_size)

    def load_mapping(self, mapping_path):
        """
        read (voacb count) file and generate self.word2idx, self.idx2word
        """
        idx2phn = self.load_pickle(mapping_path)
        self.idx2phn = {}
        for key in idx2phn.keys():
            self.idx2phn[key] = self.word2idx[idx2phn[key]]
        self.idx_size = len(self.idx2phn)

    def load_pickle(self, fpath):
        with open(fpath,'rb') as f:
            data = cPickle.load(f)
            return data

    def load_target_data(self, fpath):
        """
        read the caption file
        return:
        a list of all the caption with blank betwnn words
        """
        #sents = [regex.sub("[^\s\p{Latin}']", "", line for line in codecs.open(fpath, 'r', 'utf-8').read().split("\n") if line]
        sents = [line for line in open(fpath, 'r').read().split("\n") if line]
        sents = [sent for sent in sents if len(sent.split()) <= self.max_length]
        return sents

    def process_source_data(self, idx_data, audio2vec_data, oracle_data):
        assert (len(idx_data) == len(audio2vec_data) == len(oracle_data))
        self.data_length = len(idx_data)

        for idx in range(self.data_length):
            assert (len(idx_data[idx]) == len(audio2vec_data[idx]) == len(oracle_data[idx]))

        #generate idx
        self.idx_data = np.zeros([self.data_length, self.max_length], dtype='int32')
        for idx, idx_data_seq in enumerate(idx_data):
            self.idx_data[idx] = np.lib.pad(np.array(idx_data_seq), [0, self.max_length-len(idx_data_seq)], 'constant', constant_values=(0, 0))

        #generate audio2vec
        self.feat_dim = audio2vec_data[0].shape[-1]
        self.audio2vec_data = np.zeros([self.data_length, self.max_length, self.feat_dim], dtype='float32')
        for idx, feat in enumerate(audio2vec_data):
            self.audio2vec_data[idx] = np.lib.pad(feat, [(0, self.max_length-len(feat)),(0,0)], 'constant', constant_values=(0., 0.))

        #generate oracle
        self.oracle_data = np.zeros([self.data_length, self.max_length], dtype='int32')
        for idx, oracle_data_seq in enumerate(oracle_data):
            oracle_data_seq = [self.word2idx[one_oracle] for one_oracle in oracle_data_seq]
            self.oracle_data[idx] = np.lib.pad(np.array(oracle_data_seq), [0, self.max_length-len(oracle_data_seq)], 'constant', constant_values=(0, 0))

        self.idx_length = np.zeros([self.data_length], dtype='int32')
        for idx, idx_data_seq in enumerate(idx_data):
            self.idx_length[idx] = len(idx_data_seq)

        # self.data_length = 3000
        # self.idx_data = self.idx_data[:3000]
        # self.audio2vec_data = self.audio2vec_data[:3000]
        # self.oracle_data = self.oracle_data[:3000]
        # self.idx_length = self.idx_length[:3000]

    def process_target_data(self, target_data):
        self.target_data_length = len(target_data)

        #generate y
        self.target_data = np.zeros([self.target_data_length, self.max_length], dtype='int32')
        for idx, target_data_seq in enumerate(target_data):
            target_data_seq = [self.word2idx[one_target] for one_target in (['sil'] + target_data_seq.split() + ['sil'])]
            self.target_data[idx] = np.lib.pad(np.array(target_data_seq), [0, self.max_length-len(target_data_seq)], 'constant', constant_values=(0, 0))
        
        self.target_length = np.zeros([self.target_data_length], dtype='int32')
        for idx, target_data_seq in enumerate(target_data):
            self.target_length[idx] = len(target_data_seq.split())+2
        
        # self.target_data_length = 1000
        # self.target_data = self.target_data[-1000:]
        # self.target_length = self.target_length[-1000:]

    def get_idx_batch(self, batch_size):
        batch_idx = np.random.choice(self.data_length, batch_size, replace=False)
        return self.idx_data[batch_idx], self.idx_length[batch_idx]

    def get_target_batch(self, batch_size, noise=0):
        batch_idx = np.random.choice(self.target_data_length, batch_size, replace=False)
        target_data = self.target_data[batch_idx]
        if noise:
            noise_data = np.random.choice(self.vocab_size, (batch_size, self.max_length))
            mask = np.random.binomial(1, p=noise, size=(batch_size, self.max_length))
            target_data = np.where(mask, noise_data, target_data)

        return target_data, self.target_length[batch_idx]

    def get_audio2vec_batch(self, batch_size):
        batch_idx = np.random.choice(self.data_length, batch_size, replace=False)
        return self.audio2vec_data[batch_idx], self.idx_length[batch_idx]

    # for evaluation which need to go through all data
    def generate_batch_number(self, batch_size=128, small_batch=True) :
        if not small_batch:
            return self.data_length // batch_size
        else:
            return ((self.data_length-1) // batch_size) + 1

    def reset_batch_pointer(self, shuffle=False):
        self.pointer = 0
        self.data_order = list(range(self.data_length))
        if shuffle:
            random.shuffle(self.data_order)

    def get_batch(self, batch_size=128, small_batch=True):
        if self.data_length - self.pointer < batch_size:
            if small_batch :
                batch_idx = self.data_order[self.pointer:]
            else:
                self.update_pointer()
                batch_idx = self.data_order[self.pointer:self.pointer+batch_size]
        else:
            batch_idx = self.data_order[self.pointer:self.pointer+batch_size]
        self.update_pointer()

        return self.idx_data[batch_idx], self.audio2vec_data[batch_idx], self.idx_length[batch_idx], self.oracle_data[batch_idx]

    def update_pointer(self, batch_size=128):
        if self.data_length - self.pointer <= batch_size:
            self.pointer = 0
        else:
            self.pointer += batch_size




