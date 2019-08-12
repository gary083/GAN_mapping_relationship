import tensorflow as tf
import numpy as np
from lib.discriminator import lrelu

def creating_embedding_matrix(vocab_size, embedding_dim, name):
    init = tf.contrib.layers.xavier_initializer()
    embedding_matrix = tf.get_variable(
        name=name,
        shape=[vocab_size, embedding_dim],
        initializer=init,
        trainable = True,
        dtype=tf.float32
    )
    return embedding_matrix

def idx2phn_distribution(idx, idx_sequence_length, idx_size, phn_size, reuse=False):
    with tf.variable_scope("phn_distribution") as scope:
        if reuse:
            scope.reuse_variables()
        emb = creating_embedding_matrix(idx_size, phn_size, 'log_phn_distribution')

        log_phn_distribution = tf.nn.embedding_lookup(emb, idx)
        phn_distribution = gumbel_softmax(log_phn_distribution,hard=True)
        #mask
        mask = tf.sequence_mask(idx_sequence_length, tf.shape(idx)[1])
        mask = tf.tile(tf.expand_dims(mask,-1),[1,1,phn_size])
        paddings = tf.zeros_like(phn_distribution)
        phn_distribution = tf.where(mask, phn_distribution, paddings)

        translate_id = tf.argmax(log_phn_distribution, axis=-1)
        mapping_id = tf.argmax(emb, axis=-1)

        all_phn_distribution = tf.nn.softmax(emb)

    return phn_distribution, translate_id, mapping_id, all_phn_distribution

def sample_gumbel(shape, eps=1e-20):
    """Sample from Gumbel(0, 1)"""
    U = tf.random_uniform(shape,minval=0,maxval=1)
    return -tf.log(-tf.log(U + eps) + eps)

def gumbel_softmax_sample(logits, temperature):
    """ Draw a sample from the Gumbel-Softmax distribution """
    y = logits + sample_gumbel(tf.shape(logits))
    return tf.nn.softmax( y / temperature )

def gumbel_softmax(logits, temperature=0.9, hard=False):
    y = gumbel_softmax_sample(logits, temperature)
    if hard:
        k = tf.shape(logits)[-1]
        #y_hard = tf.cast(tf.one_hot(tf.argmax(y,1),k), y.dtype)
        y_hard = tf.cast(tf.equal(y,tf.reduce_max(y,-1,keep_dims=True)),y.dtype)
        y = tf.stop_gradient(y_hard - y) + y
    return y
