import os
from tqdm import tqdm
import numpy as np
import _pickle as cPickle

def evaluation(sess, g, args, data_loader, f_log=None, save=False):
    print('start evaluating!!!!!')
    data_loader.reset_batch_pointer()
    num_batch = data_loader.generate_batch_number()

    total_phn = 0
    total_correct = 0
    all_translate_id = [] 
    for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
        idx_data, audio2vec_data, idx_length, oracle_data = data_loader.get_batch()

        feed_dict = {
            g.inputs:idx_data,
            g.seq_length:idx_length
        }

        translate_id, mapping_id = sess.run([g.translate_id, g.mapping_id], feed_dict=feed_dict)
        phn_num, phn_correct = evaluate_result(translate_id, oracle_data, idx_length)
        total_phn += phn_num
        total_correct += phn_correct
        all_translate_id.append(translate_id)

    total_accuracy = total_correct/total_phn*100
    log = 'total correct rate: ' + str(total_accuracy)
    print(log)
    if f_log is not None:
        f_log.write(log)
        f_log.write('\n')
        
    return total_accuracy

def evaluate_result(translate_id, oracle_data, idx_length):
    phn_num = np.sum(idx_length)
    phn_correct = 0
    for batch_idx in range(len(translate_id)):
        for idx in range(idx_length[batch_idx]):
            if translate_id[batch_idx][idx] == oracle_data[batch_idx][idx]:
                phn_correct += 1

    return phn_num, phn_correct

def print_mapping_id_accuracy(mapping_id, idx2phn, f_log=None):
    count = 0
    for idx in range(len(mapping_id)):
        if mapping_id[idx] == idx2phn[idx]:
            count += 1

    accuracy = count/len(mapping_id) * 100.
    log = 'accuracy: {0:.5f}%'.format(accuracy)
    print(log)

    if f_log is not None:
        f_log.write(log)
        f_log.write('\n')
