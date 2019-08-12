import tensorflow as tf
import numpy as np

def lrelu(x, alpha=0.1):
    return tf.nn.relu(x) - alpha * tf.nn.relu(-x)

def rnn_encoder(inputs, seq_length, hidden_units):
    with tf.variable_scope("encoder"):
        init_c  = tf.get_variable(name='fw_c', shape=[hidden_units],dtype=tf.float32)
        init_h  = tf.get_variable(name='fw_h', shape=[hidden_units],dtype=tf.float32)
        init_c  = tf.tile(tf.expand_dims(init_c, axis=0),[tf.shape(inputs)[0],1])
        init_h  = tf.tile(tf.expand_dims(init_h, axis=0),[tf.shape(inputs)[0],1])
        init_fw = tf.contrib.rnn.LSTMStateTuple(c=init_c, h=init_h)
        cell_fw = tf.contrib.rnn.BasicLSTMCell(hidden_units)

        encoder_outputs, encoder_state = tf.nn.dynamic_rnn(
            cell=cell_fw,
            initial_state=init_fw,
            inputs = inputs,
            sequence_length = seq_length,
            dtype = tf.float32)

        mu     = tf.layers.dense(encoder_state.h, hidden_units, activation=lrelu, name='mu0')
        mu     = tf.layers.dense(mu, hidden_units, name='mu1')
        logvar = tf.layers.dense(encoder_state.h, hidden_units, activation=lrelu, name='logvar0')
        logvar = tf.layers.dense(logvar, hidden_units, name='logvar1')

    return mu, logvar

def rnn_decoder(z, max_length, hidden_units, feat_dim):
    init_c  = tf.get_variable(name='fw_c', shape=[hidden_units],dtype=tf.float32)
    init_h  = tf.get_variable(name='fw_h', shape=[hidden_units],dtype=tf.float32)
    init_c  = tf.tile(tf.expand_dims(init_c, axis=0),[tf.shape(z)[0],1])
    init_h  = tf.tile(tf.expand_dims(init_h, axis=0),[tf.shape(z)[0],1])
    init_state = tf.contrib.rnn.LSTMStateTuple(c=init_c, h=init_h)

    inputs = tf.tile(tf.expand_dims(z,1),[1, max_length, 1])

    with tf.variable_scope("decoder_lstm"):
        decoder_cell = tf.contrib.rnn.BasicLSTMCell(hidden_units)
        outputs, decoder_state = tf.nn.dynamic_rnn(
            cell = decoder_cell,
            inputs = inputs,
            initial_state = init_state,
            dtype = tf.float32)
    with tf.variable_scope("dense"):
        mu = tf.layers.dense(outputs, feat_dim, name='rnn_output_mu')
        logvar = tf.layers.dense(outputs, feat_dim, name='rnn_output_logvar')

    return mu, logvar

def sampling(mu, logvar):
    std = tf.exp(0.5 * logvar)
    z = mu + std * tf.random_normal(tf.shape(mu), 0, 1, dtype=tf.float32)
    return  z

def compute_kld(mu, logvar):
    kld = tf.reduce_mean(-0.5 * tf.reduce_sum(1 + logvar - tf.pow(mu,2) - tf.exp(logvar), -1))
    return kld

def seq_resonstruction_loss(inputs, reconstruction, seq_length=None):
    with tf.variable_scope('resonstruction') as scope:
        loss = tf.reduce_mean((reconstruction - inputs)**2, axis=-1)
        #masking the padding part to 1(cut and reset)
        if seq_length is not None:
            mask = tf.sequence_mask(seq_length, tf.shape(inputs)[1])
            paddings = tf.zeros_like(loss)
            loss = tf.where(mask, loss, paddings)
            mean_loss = tf.reduce_sum(loss)/tf.cast(tf.reduce_sum(seq_length),tf.float32)
        else:
            mean_loss = tf.reduce_mean(loss)
    return mean_loss

def kl_distance(mu, logvar):
	return -0.5 * tf.reduce_mean(tf.reduce_sum(1 + logvar - tf.pow(mu, 2) - tf.exp(logvar), reduction_indices=1))


