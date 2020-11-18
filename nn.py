import tensorflow as tf
import tensorflow.keras as keras


def gcn(X, A, W, b=None, act="relu"):  # node_size, embed_size;node_size, node_size;embed_size, unit;u
    d = tf.linalg.diag(tf.pow(tf.reduce_sum(A, axis=-1), -0.5))
    A_ = tf.matmul(tf.transpose(tf.matmul(A, d)), d)
    output = tf.matmul(A_, X)  # node_size, embed_size
    output = tf.matmul(output, W)  # node_size, unit
    if b is not None:
        output = tf.nn.bias_add(output, b)
    if act is not None:
        output = keras.activations.get(act)(output)
    return output
