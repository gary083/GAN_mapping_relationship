import os
import _pickle as cPickle

if __name__ == '__main__':
    with open('./phones.60-48-39.map.txt', 'r') as f:
        all_line = f.read().splitlines()
    all_phn_39 = list(set([line.split()[-1] for line in all_line]))
    assert(len(all_phn_39) == 39)

    #generate all phn file
    if not os.path.exists('./preprocessed'):
        os.mkdir('preprocessed')
    path = os.path.join('./preprocessed', 'all_vocab.txt')
    with open(path, 'w') as f:
        for word in all_phn_39:
            f.write(word)
            f.write('\n')

