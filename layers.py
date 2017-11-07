"""
Implementation of EDSR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

from tensorflow.python.framework import ops
from tensorflow.python.ops import math_ops


# subpixel shuffle
def subpixel(x, factor):

    # def _phase_shift(x, factor):
    """
    Perform the phase shift of x, the operation will transform
    a tensor of size (batch_size, height, width, input_channel) to
    (batch_size, height*factor, width*factor, 1).
    Note that input_channel = factor x factor and output_channel = 1
    Input:
        x - 4D tensor [batch_size, height, width, input_channel]
        factor - integer indicator the factor of superresolution
    Output:
    out - 4D tensor [batch_size, height*factor, width*factor, output_channel]

    Ref: 
        - `Real-Time Single Image and Video Super-Resolution Using an Efficient
            Sub-Pixel Convolutional Neural Network
        - tensorlayer
        - https://github.com/zsdonghao/tensorlayer/blob/ef533699622ab124e28526bbe97cb2edd01e54b8/tensorlayer/layers.py#L2238
    """
    batch_size, height, width, input_channel = x.get_shape().as_list()
    batch_size = tf.shape(x)[0]
    list_x = tf.split(x, factor, axis=3)
    x = tf.concat(list_x, axis=2)
    output_channel = int(input_channel / factor**2)
    x = tf.reshape(x, [batch_size, height * factor,
                       width * factor, output_channel])
    return x

# resnet block - EDSR style
def ResBlock_EDSR(x, n_filters, multiplier=0.01, name_scope='Resblock'):
    """
    Define a ResBlock specified by the EDSR paper. This is DIFFERENT
    from the conventional ResBlock
    """
    with tf.variable_scope(name_scope):

        with tf.variable_scope('Conv_1'):
            weights = tf.get_variable("weights", shape=[3, 3, n_filters, n_filters],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable('bias', shape=[n_filters],
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x, weights, strides=[
                1, 1, 1, 1], padding='SAME')
            net = tf.add(conv, bias, name='Add')
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('bias', bias)

        with tf.variable_scope('Relu'):
            act = tf.nn.relu(net)
            net = act
            tf.summary.histogram('relu', act)

        with tf.variable_scope('Conv_2'):
            weights = tf.get_variable("weights", shape=[3, 3, n_filters, n_filters],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable('bias', shape=[n_filters],
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(net, weights, strides=[
                1, 1, 1, 1], padding='SAME')
            net = tf.add(conv, bias, name='Add')
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('bias', bias)
        with tf.variable_scope('Elementwise_add'):
            net = tf.scalar_mul(multiplier, net)
            net = tf.add(x, net, name='Add')

    return net


# Leaky ReLU unit tf r1.4 implementation
def Leaky_relu(x, alpha=0.1, name=None):
    with ops.name_scope(name, "LeakyRelu", [x, alpha]):
        x = ops.convert_to_tensor(x, name="x")
        alpha = ops.convert_to_tensor(alpha, name="alpha")
        return math_ops.maximum(alpha * x, x)

# Convolution Block
def Conv2d(x, n_filters_in, n_filters_out, stride=1, bn=False, is_train=True,
           kernel_size=3, name_scope='Conv_1',activiation=None):
    with tf.variable_scope(name_scope):
        weights = tf.get_variable("weights", shape=[
            kernel_size, kernel_size, n_filters_in, n_filters_out],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", shape=[n_filters_out],
                               initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, weights, strides=[
            1, stride, stride, 1], padding='SAME')
        net = tf.add(conv, bias, name='Add')
        # if bn == True:
        #     scale = tf.get_variable("scale", shape=[n_filters_out])
        #     offset = tf.get_variable("offset", shape=[n_filters_out])
        #     net = tf.nn.fused_batch_norm(net, scale, offset, is_training=is_train)
        if activiation == 'relu':
            act = tf.nn.relu(net, name='ReLU')
            tf.summary.histogram('relu', act)
            net = act
        if activiation == 'lrelu':
            act = Leaky_relu(net, alpha=0.1, name='LReLU')
            tf.summary.histogram('lrelu', act)
            net = act
        if activiation == 'tanh':
            act = tf.nn.tanh(net, name='Tanh')
            tf.summary.histogram('tanh', act)
            net = act
        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)
        return net

# Upsampler
def Upsampler(x, n_filters_in, name_scope='Upsampler'):

    with tf.variable_scope(name_scope):
        with tf.variable_scope('Conv'):
            weights = tf.get_variable("weights", shape=[3, 3, n_filters_in, n_filters_in],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable("bias", shape=[n_filters_in],
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(x, weights, strides=[
                                1, 1, 1, 1], padding='SAME')
            net = tf.add(conv, bias, name='Add')
            # NOTE: ACTIVATION IS REQUIRED HERE
            act = tf.nn.relu(net, name='relu')
            net = act
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('bias', bias)
            tf.summary.histogram('relu', act)
        with tf.variable_scope('Subpixel_shuffle'):
            net = subpixel(net, factor=2)
    return net


def Dense(x, output_channel,
    activiation='none', name_scope='Dense'):
    input_channel = x.get_shape()[-1]
    with tf.variable_scope(name_scope):
        weights = tf.get_variable("weights",
            shape=[input_channel, output_channel],
            initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", shape=[output_channel])
        net = tf.matmul(x, weights) + bias
        if activiation=='lrelu':
            act = tf.nn.relu(net, name='LReLU')
            tf.summary.histogram('lrelu', act)
            net = act
        return net
