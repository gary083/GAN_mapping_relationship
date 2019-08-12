from evaluation import *

import numpy as np
import os
import tensorflow as tf

def train(sess, g, args, saver, data_loader):
    print ('training unsupervised')

    f_log = open(os.path.join(args.save_dir, 'log'), 'w')

    summary_dir = os.path.join(args.save_dir, 'summary')
    if not os.path.exists(summary_dir):
        os.makedirs(summary_dir)
    summary_writer = tf.summary.FileWriter(summary_dir, g.graph)
    max_acc = 0.0
    for step in list(range(1, args.step+1)):
        lr = args.discriminator_lr
        #train discriminator for several iter
        for i in range(args.discriminator_iterations):
            #sample source and real data
            x, x_len = data_loader.get_idx_batch(args.batch_size)
            y, y_len = data_loader.get_target_batch(args.batch_size)

            feed_dict = {
                g.inputs:x,
                g.seq_length:x_len,
                g.real_sample:y,
                g.real_seq_length:y_len,
                g.learning_rate:lr
            }

            _, discriminator_loss, summary, gs = sess.run([g.train_discriminator_op, g.discriminator_loss, g.discriminator_summary, g.discriminator_global_step] , feed_dict=feed_dict)

            #write summary
            summary_writer.add_summary(summary, gs)

        #train generator and reconstructor
        x, x_len = data_loader.get_idx_batch(args.batch_size)
        y, y_len = data_loader.get_target_batch(args.batch_size)

        lr = args.generator_lr

        feed_dict = {
            g.inputs:x,
            g.seq_length:x_len,
            g.real_sample:y,
            g.real_seq_length:y_len,
            g.learning_rate:lr
        }
        _, generator_loss, summary, gs = sess.run([g.train_generator_op,g.generator_loss, g.generator_summary, g.generator_global_step],feed_dict=feed_dict)

        # loss
        #write summary
        summary_writer.add_summary(summary, gs)
        #evaluate mapping id
        mapping_id = sess.run([g.mapping_id], feed_dict=feed_dict)
        
        if step % 100 == 0:
            log = 'Step: {0:3d} generator_loss: {1:5f} discriminator_loss: {2:5f}'.format(step, generator_loss, discriminator_loss)
            print (log)
            f_log.write(log)
            f_log.write('\n')
            f_log.flush()

            print_mapping_id_accuracy(mapping_id[0], data_loader.idx2phn, f_log=f_log)
        if step % 200 == 0:
            print(mapping_id[0])
            print ('evaluating step {0}'.format(step))
            acc = evaluation(sess, g, args, data_loader, f_log=f_log)
            if acc > max_acc:
                max_acc = acc
                saver.save(sess, os.path.join(args.save_dir, 'model' ))
    f_log.close()
    evaluation(sess, g, args, data_loader, f_log=open(result_file, 'w'))
