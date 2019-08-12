import os
import sys
import _pickle as cPickle
import numpy as np
import random

class data_loader:
    def __init__(self, feat_path, target_path, batch_size, meta_path=None, max_length=20, is_training=True):
        """
        Member variables:
        x: a numpy [data_length x time x MFCC]
        y: a numpy [data_length x max_len]
        self.origin_length
        self.word2idx, self.idx2word
        """
        self.is_training = is_training
        self.batch_size = batch_size
        self.max_length = max_length
        self.load_target_vocab()
        if self.is_training:
            self.load_all_speaker()

        print('loading data')
        acoustic_data = self.load_pickle(feat_path)
        target_data = self.load_pickle(target_path)
        if self.is_training:
            meta_data = self.load_pickle(meta_path)
        else:
            meta_data = None

        self.create_data(acoustic_data, target_data, meta_data)
        self.reset_batch_pointer(shuffle=False)
        print("finish loading data")

    def load_target_vocab(self):
        """
        read (voacb count) file and generate self.word2idx, self.idx2word
        """
        vocab = [line.split()[0] for line in open(os.path.join('preprocessed', 'all_vocab.txt'), 'r').read().splitlines()]
        self.word2idx = {word: idx for idx, word in enumerate(vocab)}
        self.idx2word = {idx: word for idx, word in enumerate(vocab)}
        self.vocab_size = len(self.word2idx)

    def load_all_speaker(self):
        speakers = [line.split()[0] for line in open(os.path.join('preprocessed', 'all_speaker.txt'), 'r').read().splitlines()]
        self.speaker2idx = {speaker: idx for idx, speaker in enumerate(speakers)}
        self.idx2speaker = {idx: speaker for idx, speaker in enumerate(speakers)}
        self.speaker_size = len(self.speaker2idx)

    def load_pickle(self, fpath):
        with open(fpath,'rb') as f:
            data = cPickle.load(f)
            return data

    def create_data(self, acoustic_data, target_data, meta_data):
        """
        source_mfcc_data: a list of mfcc tensor
        target_data: [[utte(seq of phns)]]
        """
        #make sure the length of mfcc and caption file are the same
        try:
            assert (len(acoustic_data) == len(target_data))
            if self.is_training:
                assert (len(acoustic_data) == len(meta_data['prefix']))
        except:
            print('the length of the mfcc and caption is not the same')
            print(len(acoustic_data), len(target_data), len(meta_data['prefix']))
            sys.exit(-1)

        feat_list       = []
        target_list     = []
        speaker_id_list = []
        sent_id_list    = []
        data_id_list   = []
        sequence_length_list = []

        sent_id = 0
        data_id = 0
        for utte_idx in range(len(acoustic_data)):
            sequence_length = 0
            if self.is_training:
                speaker_name = meta_data['prefix'][utte_idx].split('_')[1]
                speaker_id = self.speaker2idx[speaker_name]
            else:
                speaker_id = -1
            for one_vocab in target_data[utte_idx]:
                # if one_vocab[0] == 'h#':
                #     continue
                if one_vocab[1] == one_vocab[2]:
                	feat_list.append(acoustic_data[utte_idx][one_vocab[1]-1:one_vocab[2]])
                else:
                	feat_list.append(acoustic_data[utte_idx][one_vocab[1]:one_vocab[2]+1])
                target_list.append(self.word2idx[one_vocab[0]])
                speaker_id_list.append(speaker_id)
                sent_id_list.append(sent_id)
                data_id_list.append(data_id)
                sequence_length += 1 

                data_id += 1
            sent_id += 1
            sequence_length_list.append(sequence_length)

        self.data_length = len(feat_list)
        self.sent_num = sent_id
        self.feat_dim = feat_list[0].shape[-1]
 
        print('data_length: ', self.data_length)
        #generate x
        self.x = np.zeros([self.data_length, self.max_length, self.feat_dim], dtype='float32')
        for idx, feat in enumerate(feat_list):
            if len(feat) > self.max_length:
                feat = feat[0:self.max_length,:]
            self.x[idx] = np.lib.pad(feat, [(0, self.max_length-len(feat)),(0,0)], 'constant', constant_values=(0., 0.))

        #generate y
        self.y = np.array(target_list, dtype='int32')

        #generate length
        self.origin_length = np.zeros([self.data_length], dtype='int32')
        for idx, feat in enumerate(feat_list):
            length = len(feat)
            if length > self.max_length:
                length = self.max_length
            self.origin_length[idx] = length

        self.speaker_id = np.array(speaker_id_list, dtype='int32')
        self.sent_id = np.array(sent_id_list, dtype='int32')
        self.data_id = np.array(data_id_list, dtype='int32')
        self.sequence_length = np.array(sequence_length_list, dtype='int32')

    def generate_batch_number(self,small_batch=False) :
        if not small_batch:
            return self.data_length // self.batch_size
        else:
            return ((self.data_length-1) // self.batch_size) + 1

    def reset_batch_pointer(self, shuffle=True):
        self.pointer = 0
        self.data_order = list(range(self.data_length))
        if shuffle:
            random.shuffle(self.data_order)

    def get_batch(self,small_batch=False):
        if self.data_length - self.pointer < self.batch_size:
            if small_batch :
                batch_idx = self.data_order[self.pointer:]
            else:
                self.update_pointer()
                batch_idx = self.data_order[self.pointer:self.pointer+self.batch_size]
        else:
            batch_idx = self.data_order[self.pointer:self.pointer+self.batch_size]
        self.update_pointer()

        return self.x[batch_idx], self.y[batch_idx], self.origin_length[batch_idx], self.speaker_id[batch_idx] 


    def update_pointer(self):
        if self.data_length - self.pointer <= self.batch_size:
            self.pointer = 0
        else:
            self.pointer += self.batch_size
