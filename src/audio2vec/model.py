"""
https://github.com/hwalsuklee/tensorflow-mnist-AAE/blob/master/aae.py
"""
import tensorflow as tf
from module import *
from vae_module import *

class model():
    def __init__(self, args, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            #placeholder
            self.inputs = tf.placeholder(tf.float32, shape=[None, args.max_length, args.feat_dim])
            self.seq_length = tf.placeholder(tf.int32, shape=[None])

            # encoder
            with tf.variable_scope("encoder"):
                self.mu, self.logvar = rnn_encoder(self.inputs, self.seq_length, args.hidden_units)

            # decoder
            with tf.variable_scope("decoder"):
                self.reconstruct_mu, self.reconstruct_logvar = rnn_decoder(self.mu, args.max_length, args.hidden_units, args.feat_dim)

            #loss
            if is_training:
                self.learning_rate = tf.placeholder(tf.float32, shape=[])
                self.kl_weight   = tf.Variable(float(0), trainable=False)
                self.global_step = tf.Variable(0, name='global_step', trainable=False)

                self.kl_weight_op = self.kl_weight.assign(1 -2/(tf.exp(args.kl_saturate*tf.cast(self.global_step, tf.float32)/args.kl_step) + tf.exp(-args.kl_saturate*tf.cast(self.global_step, tf.float32)/args.kl_step)))

                # reconstruct Loss
                self.reconstruct_loss = seq_resonstruction_loss(self.inputs, self.reconstruct_mu, seq_length=self.seq_length)
                self.kl_distance_loss = kl_distance(self.mu, self.logvar)

                self.total_loss = self.reconstruct_loss + self.kl_distance_loss * self.kl_weight

                reconstruct_summary = tf.summary.scalar("reconstruct_loss", self.reconstruct_loss)
                kl_distance_summary = tf.summary.scalar("kl_distance_loss", self.kl_distance_loss)
                kl_weight_summary = tf.summary.scalar("kl_weight", self.kl_weight)

                self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
                gradients, variables = zip(*self.optimizer.compute_gradients(self.total_loss))
                gradients, _ = tf.clip_by_global_norm(gradients, 10.0)
                self.train_op = self.optimizer.apply_gradients(zip(gradients, variables), global_step=self.global_step)

                # final summary operations
                self.reconstruct_summary = tf.summary.merge([reconstruct_summary])
                self.kl_distance_summary = tf.summary.merge([kl_distance_summary])
                self.kl_weight_summary   = tf.summary.merge([kl_weight_summary])


