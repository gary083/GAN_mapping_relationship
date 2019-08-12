import os
import _pickle as cPickle
from collections import Counter

def make_speaker(meta_path):
    meta = cPickle.load(open(meta_path, 'rb'))
    all_prefix = meta['prefix']
    all_speaker = [prefix.split('_')[1] for prefix in all_prefix]
    return Counter(all_speaker)

def make_phn(phn_path):
    text = cPickle.load(open(phn_path, 'rb'))
    all_vocab = []
    for utte in text:
        for vocab in utte:
            if vocab[0] != 'h#':
                all_vocab.append(vocab[0])
    all_vocab = Counter(all_vocab)
    return all_vocab

def output2file(path, name2cnt):
     with open(path, 'w') as f:
        for word in name2cnt.keys():
            f.write("{}\t{}\n".format(word, name2cnt[word]))

if __name__ == '__main__':
    if not os.path.exists('./preprocessed'): 
        os.mkdir('preprocessed')
    root = '/home/guanyu/guanyu/timit/'
    #generate all speaker file
    train_meta_path = os.path.join(root, 'timit-train-meta.pkl')
    speaker2cnt = make_speaker(train_meta_path)
    path = os.path.join('./preprocessed', 'all_speaker.txt')
    output2file(path, speaker2cnt)
    print('total speaker num:', len(speaker2cnt))

    #generate all phn file
    train_vocab_path = os.path.join(root, 'timit-train-phn.pkl')
    vocab2cnt = make_phn(train_vocab_path)
    path = os.path.join('./preprocessed', 'all_vocab.txt')
    output2file(path, vocab2cnt)
    print('total vocab num:', len(vocab2cnt))


