import numpy as np
import math
from model import *
from tqdm import tqdm
import os

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def train(sess, g, args, saver, data_loader):
    print ('start training')
    reconstruct_losses = AverageMeter()
    kl_distance_losses = AverageMeter()
    min_loss = np.inf

    num_batch = data_loader.generate_batch_number()

    f_log = open(os.path.join(args.save_dir, args.log), 'w')

    summary_dir = os.path.join(args.save_dir, 'summary')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)

    summary_writer = tf.summary.FileWriter(summary_dir, g.graph)

    hidden_size = args.hidden_units

    for epoch in list(range(1, args.epoch+1)):
        reconstruct_losses.reset()
        kl_distance_losses.reset()
        data_loader.reset_batch_pointer()
        # lr = args.lr*math.pow(args.decay_rate, max((epoch-args.decay_after),0))
        lr = args.lr
        
        for step in tqdm(range(num_batch), total=num_batch, ncols=70, leave=False, unit='b'):
            x, phn_idx, x_length, _ = data_loader.get_batch()

            #train reconstructor
            feed_dict = {
                g.inputs:x,
                g.seq_length:x_length,
                g.learning_rate:lr
            }

            _, _, reconstruct_loss, kl_distance_loss, gs = sess.run([g.train_op, g.kl_weight_op, g.reconstruct_loss, g.kl_distance_loss, g.global_step], feed_dict=feed_dict)
            kl_weight, reconstruct_summary, kl_distance_summary, kl_weight_summary = sess.run([g.kl_weight, g.reconstruct_summary, g.kl_distance_summary, g.kl_weight_summary], feed_dict=feed_dict)
            #update
            reconstruct_losses.update(reconstruct_loss, np.sum(x_length))
            kl_distance_losses.update(kl_distance_loss, np.sum(x_length))
            #write summary
            summary_writer.add_summary(reconstruct_summary, gs)
            summary_writer.add_summary(kl_distance_summary, gs)
            summary_writer.add_summary(kl_weight_summary,   gs)

        #save
        if min_loss > reconstruct_losses.avg + kl_distance_losses.avg:
            gs = sess.run(g.global_step)
            min_loss = reconstruct_losses.avg + kl_distance_losses.avg
            saver.save(sess, os.path.join(args.save_dir, 'model'))
            print ("save model")

        log = 'epoch: {0:3d} reconstructor_loss: {1:5f} kl_distance_losses: {2:5f} kl_weight: {3:5f}'.format(epoch, reconstruct_losses.avg, kl_distance_losses.avg, kl_weight)
        print (log)
        f_log.write(log)
        f_log.write('\n')
        f_log.flush()

        #write summary

    f_log.close()
    print ('Finish training')
