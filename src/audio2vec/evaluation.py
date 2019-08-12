import numpy as np
from model import *
from tqdm import tqdm
import os
from sklearn import cluster
import time
import heapq
import _pickle as cPickle

class matching(object):
    def __init__(self, phn_length, embed_length):
        self.phn_length = phn_length
        self.embed_length = embed_length
        self.reset()
    def reset(self):
        self.matching_matrix = np.zeros([self.phn_length, self.embed_length], dtype=np.int32)
    def update(self, phn_idx, embed_idx):
        self.matching_matrix[phn_idx][embed_idx] += 1
    def upper_bound_classification(self):
        all_element = np.sum(self.matching_matrix)
        count = 0
        for idx in range(self.embed_length):
            count += np.amax(self.matching_matrix[:,idx])
        return count/all_element
    def max_matching_map(self):
        mapping = {}
        for idx in range(self.embed_length):
            mapping[idx] = np.argmax(self.matching_matrix[:,idx])
        return mapping

    def print_matching(self, id2word, f):
        for idx in range(self.embed_length):
            count_array = self.matching_matrix[:,idx]
            all_phn_id = heapq.nlargest(2, range(len(count_array)), count_array.take)
            num0 = count_array[all_phn_id[0]]
            num1 = count_array[all_phn_id[1]]
            total_num = np.sum(count_array)
            log = 'idx: {0:3d} phn0: {1:5s} num: {2:5d}/{3:5d} phn1: {4:5s} num: {5:5d}/{6:5d}'.format(idx, id2word[all_phn_id[0]], num0, total_num, id2word[all_phn_id[1]], num1, total_num)
            f.write(log)
            f.write('\n')
            print(log)
        all_element = np.sum(self.matching_matrix)
        log = 'total num: {0:5d}'.format(all_element)
        f.write(log)
        f.write('\n')
        print(log)

def evaluation(sess, g, args, data_loader_train, data_loader_test):
    print('start evaluating!!!!!')
    train_all_vector, train_all_phn_label, train_all_speaker_id = get_all_result(sess, g, data_loader_train)
    test_all_vector, test_all_phn_label, test_all_speaker_id = get_all_result(sess, g, data_loader_test)

    #read 60-39 phn pair
    phn_60_to_39, idx2phn_39, phn2idx_39 = read_phn_map('./phones.60-48-39.map.txt')

    #train
    # 39 phn
    train_all_phn_label_39 =  match_to_39_phn(train_all_phn_label, phn_60_to_39, phn2idx_39, data_loader_train.idx2word)
    train_upper_bound, train_cluster_id, kmeans_fit, best_map = count_upper_bound(train_all_vector, train_all_phn_label_39, 
                                                39, cluster_num=args.cluster_num, idx2phn=idx2phn_39, 
                                                matching_path=os.path.join(args.save_dir, 'matching_result_39'))
    print('39 before subtraction: ', train_upper_bound)

    #test
    test_cluster_id = generate_cluster_id(test_all_vector, kmeans_fit)
    test_all_phn_label_39 =  match_to_39_phn(test_all_phn_label, phn_60_to_39, phn2idx_39, data_loader_test.idx2word)
    test_accuracy = get_accuracy(test_cluster_id, test_all_phn_label_39, best_map)
    print('test accuracy: ', test_accuracy)
    if args.output:
        save_dir = args.cluster_dir+"/"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        # save train file
        save_data(save_dir+'train', train_all_vector, train_all_phn_label_39, train_cluster_id, data_loader_train.sequence_length, idx2phn_39)
        #save test file
        save_data(save_dir+'test', test_all_vector, test_all_phn_label_39, test_cluster_id, data_loader_test.sequence_length, idx2phn_39)

        save_map_file(save_dir+'best', best_map, idx2phn_39)


def get_all_result(sess, g, data_loader):
    data_loader.reset_batch_pointer(shuffle=False)
    num_batch = data_loader.generate_batch_number(small_batch=True)

    all_vector = []
    all_phn_label = []
    all_speaker_id = []
    for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        x, phn_idx, x_length, speaker_id = data_loader.get_batch(small_batch=True)

        feed_dict = {
            g.inputs:x,
            g.seq_length:x_length
        }

        enc = sess.run([g.mu], feed_dict=feed_dict)
        all_vector.append(enc[0])
        all_phn_label.append(phn_idx)
        all_speaker_id.append(speaker_id)

    all_vector = np.concatenate(all_vector, axis=0)
    all_phn_label = np.concatenate(all_phn_label, axis=0)
    all_speaker_id = np.concatenate(all_speaker_id, axis=0)

    assert(len(all_vector) == len(all_phn_label) == data_loader.data_length)

    return all_vector, all_phn_label, all_speaker_id


def count_upper_bound(all_vector, all_phn, phn_size, cluster_num=300, idx2phn=None, matching_path=None):
    cluster_id, kmeans_fit = cluster_by_kmeans(all_vector, cluster_num=cluster_num)
    assert(len(all_vector) == len(all_phn) == len(cluster_id))

    match = matching(phn_size, cluster_num)
    match.reset()
    for idx in range(len(all_vector)):
        match.update(all_phn[idx], cluster_id[idx])
    # print matching result
    if idx2phn is not None:
        match.print_matching(idx2phn, open(matching_path,'w'))

    best_map = match.max_matching_map()

    return match.upper_bound_classification() * 100, cluster_id, kmeans_fit, best_map

def cluster_by_kmeans(all_vector, cluster_num=300):
    print('kmeans clustering')
    start = time.time()
    kmeans_fit = cluster.KMeans(n_clusters=cluster_num).fit(all_vector)
    print('finish clustering')
    print('kmeans time: ', (time.time()-start)/60, 'minute')
    return kmeans_fit.labels_, kmeans_fit

def read_phn_map(path):
    all_lines = open(path, 'r').read().splitlines()
    phn_mapping = {}
    for line in all_lines:
        if line.strip() == "":
            continue
        phn_mapping[line.split()[0]] = line.split()[2]

    #create the transformation between id and phn_39
    all_39_phn = list(set(phn_mapping.values()))
    assert(len(all_39_phn) == 39)
    phn2idx_39 = dict(zip(all_39_phn, range(len(all_39_phn))))
    idx2phn_39 = dict(zip(range(len(all_39_phn)), all_39_phn))

    return phn_mapping, idx2phn_39, phn2idx_39

def match_to_39_phn(all_phn, phn_60_to_39, phn2idx_39, idx2phn):
    all_phn_39 = np.zeros_like(all_phn)
    for idx in range(len(all_phn)):
        phn_60_idx = all_phn[idx]
        phn_60 = idx2phn[phn_60_idx]
        all_phn_39[idx] = phn2idx_39[phn_60_to_39[phn_60]]

    return all_phn_39

def generate_cluster_id(all_vector, kmeans_fit):
    return kmeans_fit.predict(all_vector)

def get_accuracy(cluster_id, all_phn_label_39, best_map):
    assert(len(cluster_id) == len(all_phn_label_39))
    all_num = len(cluster_id)
    correct = 0
    for idx in range(len(cluster_id)):
        if best_map[cluster_id[idx]] == all_phn_label_39[idx]:
            correct += 1
    return correct/all_num * 100

def save_data(save_prefix, all_vector, all_phn_label_39, cluster_id, sequence_length, idx2phn):
    assert(len(all_vector) == len(all_phn_label_39) == len(cluster_id) == np.sum(sequence_length))
    start = 0

    all_audio2vec = []
    all_phn_label = []
    all_cluster_id = []

    for idx in range(len(sequence_length)):
        length = sequence_length[idx]
        #audio2vec
        audio2vec = all_vector[start:start+length]
        all_audio2vec.append(audio2vec)

        #phn label
        phn_label_temp = []
        for phn_idx in range(start, start+length):
            phn_label_temp.append(idx2phn[all_phn_label_39[phn_idx]])
        all_phn_label.append(phn_label_temp)

        #cluster_id
        cluster_id_temp = []
        for cluster_idx in range(start, start+length):
            cluster_id_temp.append(cluster_id[cluster_idx])
        all_cluster_id.append(cluster_id_temp)

        start += length

    #output
    cPickle.dump(all_audio2vec, open(save_prefix+'.audio2vec','wb'))
    cPickle.dump(all_phn_label, open(save_prefix+'.phn','wb'))
    cPickle.dump(all_cluster_id, open(save_prefix+'.cluster','wb'))
    #write_to_file(save_prefix+'.phn', all_phn_label)
    #write_to_file(save_prefix+'.cluster', all_cluster_id)

def write_to_file(path, listoflist):
    with open(path, 'w') as f:
        for one_list in listoflist:
            for element in one_list:
                f.write(str(element))
            f.write('\n')

def save_map_file(save_prefix, best_map, idx2phn_39):
    mapping = {}
    for key in best_map.keys():
        mapping[key] = idx2phn_39[best_map[key]]

    cPickle.dump(mapping, open(save_prefix+'.map','wb'))
