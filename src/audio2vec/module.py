import tensorflow as tf

def creating_embedding_matrix(vocab_size, embedding_dim, name):
    init = tf.contrib.layers.xavier_initializer()
    embedding_matrix = tf.get_variable(
        name=name,
        shape=[vocab_size, embedding_dim],
        initializer=init,
        trainable = True
    )
    return embedding_matrix

def ln(inputs, epsilon = 1e-5, scope = None):
    mean, var = tf.nn.moments(inputs, [1], keep_dims=True)
    with tf.variable_scope(scope + 'LN'):
        scale = tf.get_variable('alpha',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(1))
        shift = tf.get_variable('beta',
                                shape=[inputs.get_shape()[1]],
                                initializer=tf.constant_initializer(0))
    LN = scale * (inputs - mean) / tf.sqrt(var + epsilon) + shift

    return LN

