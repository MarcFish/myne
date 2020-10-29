import tensorflow as tf
import tensorflow.keras as keras


class DenseLayer(keras.layers.Layer):
    def __init__(self, units, dropout_prob=0.1):
        super(DenseLayer, self).__init__()
        self.layers = keras.Sequential()
        for unit in units:
            self.layers.add(keras.layers.Dense(units=unit))
            self.layers.add(keras.layers.LeakyReLU(0.2))
            self.layers.add(keras.layers.LayerNormalization())
            self.layers.add(keras.layers.Dropout(dropout_prob))

    def call(self, inputs):
        o = self.layers(inputs)
        return o


class ResidualLayer(keras.layers.Layer):
    def __init__(self, unit1s, unit2s, dropout_prob=0.1):
        super(ResidualLayer, self).__init__()
        self.layer1 = keras.Sequential()
        self.layer2 = keras.Sequential()
        self.unit1s = unit1s
        self.unit2s = unit2s
        for unit in unit1s:
            self.layer1.add(keras.layers.Dense(units=unit))
            self.layer1.add(keras.layers.LeakyReLU(0.2))
            self.layer1.add(keras.layers.LayerNormalization())
            self.layer1.add(keras.layers.Dropout(dropout_prob))

        for unit in unit2s:
            self.layer2.add(keras.layers.Dense(units=unit))
            self.layer2.add(keras.layers.LeakyReLU(0.2))
            self.layer2.add(keras.layers.LayerNormalization())
            self.layer2.add(keras.layers.Dropout(dropout_prob))
        self.leakyrelu = keras.layers.LeakyReLU(0.2)
        self.ln = keras.layers.LayerNormalization()
        self.drop = keras.layers.Dropout(dropout_prob)

    def build(self, input_shape):
        if input_shape[-1] != self.unit2s[-1]:
            raise Exception("Dim not equal")
        self.built = True

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        outputs = self.leakyrelu(x + inputs)
        outputs = self.ln(outputs)
        outputs = self.drop(outputs)
        return outputs


class GraphAttention(keras.layers.Layer):
    def __init__(self, feature_size, attn_heads=8, dropout_prob=0.5, activation="relu"):
        super(GraphAttention, self).__init__()
        self.feature_size = feature_size
        self.attn_heads = attn_heads
        self.activation = keras.activations.get(activation)
        self.dropout_prob = dropout_prob

        self.kernels = list()
        self.biases = list()
        self.attn_kernels = list()

    def build(self, input_shape):  # X, A
        input_feature_size = input_shape[0][-1]
        for head in range(self.attn_heads):
            kernel = self.add_weight(shape=(input_feature_size, self.feature_size))
            self.kernels.append(kernel)
            bias = self.add_weight(shape=(self.feature_size,))
            self.biases.append(bias)
            attn_kernel_self = self.add_weight(shape=(self.feature_size, 1))
            attn_kernel_neighs = self.add_weight(shape=(self.feature_size, 1))
            self.attn_kernels.append([attn_kernel_self, attn_kernel_neighs])

        self.built = True

    def call(self, inputs):  # X, A
        X = inputs[0]
        A = inputs[1]
        outputs = list()

        for head in range(self.attn_heads):
            kernel = self.kernels[head]
            attention_kernel = self.attn_kernels[head]

            features = tf.matmul(X, kernel)

            attn_for_self = tf.matmul(features, attention_kernel[0])
            attn_for_neighs = tf.matmul(features, attention_kernel[1])

            dense = attn_for_self + tf.transpose(attn_for_neighs)
            dense = keras.layers.LeakyReLU(0.2)(dense)
            mask = (1.0 - A) * (-10e9)
            dense += mask
            dense = tf.nn.softmax(dense)

            # Mask values before activation (Vaswani et al., 2017)
            mask = -10e9 * (1.0 - A)
            dense += mask

            dense = tf.nn.softmax(dense)
            dropout_attn = tf.keras.layers.Dropout(self.dropout_prob)(dense)
            dropout_feat = tf.keras.layers.Dropout(self.dropout_prob)(features)

            node_features = tf.matmul(dropout_attn, dropout_feat)

            node_features = tf.nn.bias_add(node_features, self.biases[head])

            # Add output of attention head to final output
            outputs.append(node_features)

        output = tf.reduce_mean(tf.stack(outputs), axis=0)

        output = self.activation(output)
        return output


class GraphConvolution(keras.layers.Layer):
    """Basic graph convolution layer as in https://arxiv.org/abs/1609.02907"""
    def __init__(self, units, activation='relu', use_bias=True):
        super(GraphConvolution, self).__init__()
        self.units = units
        self.activation = keras.activations.get(activation)
        self.use_bias = use_bias

    def build(self, input_shapes):  # X, A
        features_shape = input_shapes[0]
        adjoint_shape = input_shapes[1]  # support, node_size, node_size
        assert len(features_shape) == 2
        input_dim = features_shape[1]
        support = adjoint_shape[0]
        self.support = support
        self.kernel = self.add_weight(shape=(input_dim * support, self.units))
        if self.use_bias:
            self.bias = self.add_weight(shape=(self.units,))
        self.built = True

    def call(self, inputs):  # X, A
        features = inputs[0]  # node_size, feautre_size
        basis = inputs[1]  # support, node_size, node_size

        output = tf.matmul(basis, features)  # support, node_size, feature_size
        # os = list()
        # for i in range(self.support):
        #     os.append(output[i])
        # output = tf.concat(os, axis=1)
        output = tf.reshape(output, [tf.shape(basis)[1], -1])
        output = tf.matmul(output, self.kernel)
        if self.bias:
            output += self.bias
        return self.activation(output)


class GCNFilter(keras.layers.Layer):
    def __init__(self, mode="localpool", support=1):
        super(GCNFilter, self).__init__()
        self.support = support
        if mode == "localpool":
            self.process = self._localpool
            assert support >= 1
        else:
            self.process = self._chebyshev
            assert support >= 2

    def build(self, input_shape):
        self.shape = input_shape[0]
        self.built = True

    def call(self, inputs):
        return self.process(inputs)

    def _localpool(self, inputs):
        d = tf.linalg.diag(tf.pow(tf.reduce_sum(inputs, axis=-1), -0.5))
        out = tf.matmul(tf.transpose(tf.matmul(inputs, d)), d)
        out = tf.stack([out])
        return out

    def _chebyshev(self, inputs):
        d = tf.linalg.diag(tf.pow(tf.reduce_sum(inputs, axis=-1), -0.5))
        adj_norm = tf.matmul(tf.transpose(tf.matmul(inputs, d)), d)
        laplacian = tf.eye(self.shape) - adj_norm
        largest_eigval = tf.math.reduce_max(tf.linalg.eigvalsh(laplacian))
        scaled_laplacian = (2. / largest_eigval) * laplacian - tf.eye(self.shape)
        out = list()
        out.append(tf.eye(self.shape))
        out.append(scaled_laplacian)
        for i in range(2, self.support+1):
            o = 2 * tf.matmul(scaled_laplacian, out[-1]) - out[-2]
            out.append(o)
        out = tf.stack(out)
        return out