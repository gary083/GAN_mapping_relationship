from lib.discriminator import *
from module import *

import tensorflow as tf
import numpy as np
import os

class model():
    def __init__(self, args, is_training=True):
        self.graph = tf.Graph()
        with self.graph.as_default():
            with tf.variable_scope("input") as scope:
                self.inputs = tf.placeholder(tf.int32, shape=[None, args.max_length])
                self.seq_length = tf.placeholder(tf.int32, shape=[None])
                self.real_sample = tf.placeholder(dtype=tf.int32, shape=[None, args.max_length])
                self.real_seq_length = tf.placeholder(tf.int32, shape=[None])

            # embedding
            with tf.variable_scope("generator"):
                self.false_sample, self.translate_id, self.mapping_id, self.all_phn_distribution = idx2phn_distribution(self.inputs, self.seq_length, args.idx_size, args.phn_size)

            with tf.variable_scope("discriminator"):
                emb = creating_embedding_matrix(args.phn_size, args.discriminator_hidden_units, 'emb')
                real_sample = tf.one_hot(self.real_sample, args.phn_size, dtype=tf.float32)
                #mask
                mask = tf.sequence_mask(self.real_seq_length, tf.shape(real_sample)[1])
                mask = tf.tile(tf.expand_dims(mask,-1),[1,1,args.phn_size])
                paddings = tf.zeros_like(real_sample)
                real_sample = tf.where(mask, real_sample, paddings)

                true_sample_pred = weak_discriminator(real_sample, emb, args.discriminator_hidden_units)
                false_sample_pred = weak_discriminator(self.false_sample, emb, args.discriminator_hidden_units, reuse=True)

                alpha = tf.random_uniform(
                    shape=[tf.shape(real_sample)[0],1,1],
                    minval=0.,
                    maxval=1.)

                differences = self.false_sample - real_sample
                interpolates = real_sample + (alpha*differences)

                interpolates_sample_pred = weak_discriminator(interpolates, emb, args.discriminator_hidden_units, reuse=True)

                gradients = tf.gradients(interpolates_sample_pred,[interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1,2]))
                gradient_penalty = tf.reduce_mean((slopes-1.)**2)

            #loss
            if is_training:
                self.learning_rate = tf.placeholder(tf.float32, shape=[])

                with tf.variable_scope("discriminator_loss") as scope:
                    self.true_sample_score  = tf.reduce_mean(true_sample_pred)
                    self.false_sample_score = tf.reduce_mean(false_sample_pred)
                    self.discriminator_loss = -(self.true_sample_score - self.false_sample_score) + 10.0*gradient_penalty

                with tf.variable_scope("generator_loss") as scope:
                    self.generator_loss = self.true_sample_score - self.false_sample_score

                self.discriminator_variables = [v for v in tf.trainable_variables() if v.name.startswith("discriminator")]
                self.generator_variables = [v for v in tf.trainable_variables() if v.name.startswith("generator")]

                #discriminator optimizer
                self.discriminator_global_step = tf.Variable(0, name='discriminator_global_step', trainable=False)
                train_discriminator_op = tf.train.AdamOptimizer(self.learning_rate) #adam is better
                gradients, variables = zip(*train_discriminator_op.compute_gradients(self.discriminator_loss,\
                    var_list=self.discriminator_variables))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_discriminator_op = train_discriminator_op.apply_gradients(zip(gradients, variables), global_step=self.discriminator_global_step)

                #generator optimizer
                self.generator_global_step = tf.Variable(0, name='generator_global_step', trainable=False)
                train_generator_op = tf.train.AdamOptimizer(self.learning_rate)
                gradients, variables = zip(*train_generator_op.compute_gradients(self.generator_loss,\
                    var_list=self.generator_variables))
                gradients, _ = tf.clip_by_global_norm(gradients, 5.0)
                self.train_generator_op = train_generator_op.apply_gradients(zip(gradients, variables), global_step=self.generator_global_step)

                #write summary
                discriminator_summary = tf.summary.scalar("discriminator_loss", self.discriminator_loss)
                generator_summary = tf.summary.scalar("generator_loss", self.generator_loss)
                self.discriminator_summary = tf.summary.merge([discriminator_summary])
                self.generator_summary = tf.summary.merge([generator_summary])
