import tensorflow as tf
import os

import data_load
from model import model
from train import train
from evaluation import evaluation
import argparse

def addParser():
    parser = argparse.ArgumentParser(description='Phoneme VAE translation')
    parser.add_argument('--mode',type=str, default='train',
        metavar='<mode>',
        help='training or testing')
    parser.add_argument('--load',type=str, default='not_load',
        metavar='<load_model>',
        help='whether to load model or not')
    parser.add_argument('--output', action='store_true',
        help='<--save evaluate result>')
    #about training
    parser.add_argument('--lr',  type=float, default=0.001,
        metavar='<--learning rate>')
    parser.add_argument('--max_length',type=int, default=20,
        metavar='<--max length>')
    parser.add_argument('--hidden_units',type=int, default=512,
        metavar='<--hidden dimension>',
        help='only used by rnn')

    parser.add_argument('--batch_size',type=int, default=64,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--epoch',type=int, default=100,
        metavar='--<# of epochs for training>',
        help='The number of epochs for training')

    # about kl_loss
    parser.add_argument('--kl_saturate',  type=int, default=2.5,
        metavar='<--decay_after>')
    parser.add_argument('--kl_step',  type=int, default=20000,
        metavar='<--decay_after>')

    #about path
    parser.add_argument('--log',type=str,
        metavar='<log file path>')
    parser.add_argument('--save_dir',type=str,
        metavar='<save model>')
    parser.add_argument('--cluster_dir',type=str,
        metavar='<save cluster data>')
    parser.add_argument('--train_feat',type=str,
        metavar='<acoustic training feature>')
    parser.add_argument('--test_feat',type=str,
        metavar='<acoustic testing feature>')
    parser.add_argument('--train_phn',type=str,
        metavar='<training phoneme position>')
    parser.add_argument('--test_phn',type=str,
        metavar='<testing phoneme position>')
    parser.add_argument('--meta',type=str,
        metavar='<meta file path>')

    parser.add_argument('--cluster_num',type=int, default=300,
        metavar='--<cluster_num>')

    return parser

def main(args):
    #load data
    if args.mode == 'train':
        data_loader_train = data_load.data_loader(args.train_feat, args.train_phn, args.batch_size, meta_path=args.meta, max_length=args.max_length, is_training=True)
    else:
        data_loader_train = data_load.data_loader(args.train_feat, args.train_phn, args.batch_size, meta_path=args.meta, max_length=args.max_length, is_training=True)
        data_loader_test = data_load.data_loader(args.test_feat, args.test_phn, args.batch_size, max_length=args.max_length, is_training=False)

    #add some feature to args
    args.feat_dim   = data_loader_train.feat_dim
    args.vocab_size = data_loader_train.vocab_size

    #build model graph
    if args.mode == 'train':
        g = model(args)
    else:
        g = model(args, is_training=False)
    print("Graph loaded")

    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)

    #create sess
    with tf.Session(graph=g.graph) as sess:
        sess.run(tf.global_variables_initializer())
        saver = tf.train.Saver(max_to_keep=3)

        if (args.mode != 'train') or (args.load == 'load'):
            print('load_model')
            saver.restore(sess, tf.train.latest_checkpoint(args.save_dir))

        if args.mode == 'train':
            print('training')
            train(sess, g, args, saver, data_loader_train)
        else:
            print('evaluating')
            evaluation(sess, g, args, data_loader_train, data_loader_test)

if __name__ == '__main__':
    parser = addParser()
    args = parser.parse_args()
    main(args)
