import tensorflow as tf
import argparse
import data_load
from train import *
from evaluation import *
import os
from  model import *

def addParser():
    parser = argparse.ArgumentParser(description='mapping network')
    parser.add_argument('--mode',type=str, default='train',
        metavar='<mode>',
        help='training or testing')
    parser.add_argument('--load',type=str, default='not_load',
        metavar='<load_model>',
        help='whether to load model or not')

    #about training
    parser.add_argument('--generator_lr',  type=float, default=0.01,
        metavar='<--generator learning rate>')
    parser.add_argument('--discriminator_lr',  type=float, default=0.001,
        metavar='<--discriminator learning rate>')
    parser.add_argument('--max_length',type=int, default=80,
        metavar='--<max length>')
    parser.add_argument('--batch_size',type=int, default=64,
        metavar='--<batch size>',
        help='The batch size while training')
    parser.add_argument('--step',type=int, default=10000,
        metavar='--<# of epochs for training>',
        help='The number of epochs for training')

    parser.add_argument('--discriminator_hidden_units',type=int, default=256,
        metavar='--<discriminator_hidden_units>')
    parser.add_argument('--discriminator_iterations',type=int, default=5,
        metavar='--<discriminator_iterations>')

    #about path
    parser.add_argument('--result_file',type=str,
        metavar='<result file path>')
    parser.add_argument('--save_dir',type=str,
        metavar='<save model>')

    parser.add_argument('--train_idx',type=str,
        metavar='<training index feature>')
    parser.add_argument('--train_audio2vec',type=str,
        metavar='<training audio2vec feature>')
    parser.add_argument('--train_oracle',type=str,
        metavar='<training phoneme label>')

    parser.add_argument('--test_idx',type=str,
        metavar='<testing index feature>')
    parser.add_argument('--test_audio2vec',type=str,
        metavar='<testing audio2vec feature>')
    parser.add_argument('--test_oracle',type=str,
        metavar='<testing phoneme labels>')

    parser.add_argument('--mapping',type=str,
        metavar='<mapping file>')
    parser.add_argument('--target',type=str,
        metavar='<training phoneme position>')

    return parser


def main(args):
    #load data
    if args.mode == 'train':
        data_loader = data_load.data_loader(args.max_length, args.train_idx, args.train_audio2vec, args.train_oracle, args.mapping, target_data_path=args.target)
    else:
        data_loader = data_load.data_loader(args.max_length, args.test_idx, args.test_audio2vec, args.test_oracle, args.mapping, target_data_path=None)

    #add some feature to args
    args.idx_size = data_loader.idx_size
    args.phn_size = data_loader.vocab_size
    args.feat_dim = data_loader.feat_dim

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
            train(sess, g, args, saver, data_loader)
        else:
            print('evaluating')
            evaluation(sess, g, args, data_loader)

if __name__ == "__main__":
    parser = addParser()
    args = parser.parse_args()
    main(args)
