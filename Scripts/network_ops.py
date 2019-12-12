import tensorflow as tf, numpy as np
import tensorflow.contrib.slim as slim

def conv_layer(input, filter, kernel, stride, padding='SAME', layer_name="conv"):
    with tf.name_scope(layer_name):
        # network = tf.layers.conv2d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        network = tf.layers.conv1d(inputs=input, use_bias=False, filters=filter, kernel_size=kernel, strides=stride, padding=padding)
        return network

def Linear(x) :
    return tf.layers.dense(inputs=x, activation=None, use_bias=False, units=1, name='linear')

def Global_Average_Pooling(x):
    return tf.reduce_mean(x, axis=[1, 2], name='Global_avg_pooling')

def Average_pooling(x, pool_size=2, stride=2, padding='SAME'):
    # return tf.layers.average_pooling2d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)
    return tf.layers.average_pooling1d(inputs=x, pool_size=pool_size, strides=stride, padding=padding)

def Concatenation(layers) :
    return tf.concat(layers, axis=2)
    # return tf.concat(layers, axis=3)

def Layer_Normalization(x, scope):
    return slim.layer_norm(x, scope=scope)

class layers():
    def __init__(self, depth, cardinality, blocks):
        self.depth = depth
        self.cardinality = cardinality
        self.blocks = blocks

    def first_layer(self, x, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=64, kernel=3, stride=1, layer_name=scope + '_conv1')
            x = Layer_Normalization(x, scope=scope + '_batch1')
            x = tf.nn.selu(x)

            return x

    def transform_layer(self, x, stride, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=self.depth, kernel=1, stride=stride, layer_name=scope + '_conv1')
            x = Layer_Normalization(x, scope=scope + '_batch1')
            x = tf.nn.selu(x)

            x = conv_layer(x, filter=self.depth, kernel=3, stride=1, layer_name=scope + '_conv2')
            x = Layer_Normalization(x, scope=scope + '_batch2')
            x = tf.nn.selu(x)
            return x

    def transition_layer(self, x, out_dim, scope):
        with tf.name_scope(scope):
            x = conv_layer(x, filter=out_dim, kernel=1, stride=1, layer_name=scope + '_conv1')
            x = Layer_Normalization(x, scope=scope + '_batch1')
            # x = tf.nn.selu(x)

            return x

    def split_layer(self, input_x, stride, layer_name):
        with tf.name_scope(layer_name):
            layers_split = list()
            for i in range(self.cardinality):
                splits = self.transform_layer(input_x, stride=stride, scope=layer_name + '_splitN_' + str(i))
                layers_split.append(splits)

            return Concatenation(layers_split)

    def residual_layer(self, input_x, out_dim, layer_num):
        # split + transform(bottleneck) + transition + merge

        for i in range(self.blocks):
            # input_dim = input_x.get_shape().as_list()[-1]
            input_dim = int(np.shape(input_x)[-1])

            if input_dim * 2 == out_dim:
                flag = True
                stride = 2
                channel = input_dim // 2
            else:
                flag = False
                stride = 1
            x = self.split_layer(input_x, stride=stride, layer_name='split_layer_' + layer_num + '_' + str(i))
            x = self.transition_layer(x, out_dim=out_dim, scope='trans_layer_' + layer_num + '_' + str(i))

            if flag is True:
                pad_input_x = Average_pooling(input_x)
                # pad_input_x = tf.pad(pad_input_x,
                #                      [[0, 0], [0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]
                pad_input_x = tf.pad(pad_input_x,
                                     [[0, 0], [0, 0], [channel, channel]])  # [?, height, width, channel]

            else:
                pad_input_x = input_x

            input_x = tf.nn.selu(x + pad_input_x)

        return input_x
