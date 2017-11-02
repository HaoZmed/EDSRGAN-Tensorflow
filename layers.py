"""
Implementation of EDSR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# subpixel shuffle


def subpixel(x, factor):

    def _phase_shift(x, factor):
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

    return _phase_shift(x, factor)


# resnet block - EDSR style
def ResBlock_EDSR(x, n_filters, multiplier=0.1, name_scope='Resblock'):
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
            bias = tf.get_variable('bias', shape=[n_filters],
                                   initializer=tf.constant_initializer(0.0))
            act = tf.nn.relu(tf.add(net, bias), name='Activation')
            tf.summary.histogram('relu', act)

        with tf.variable_scope('Conv_2'):
            weights = tf.get_variable("weights", shape=[3, 3, n_filters, n_filters],
                                      initializer=tf.truncated_normal_initializer(stddev=0.1))
            bias = tf.get_variable('bias', shape=[n_filters],
                                   initializer=tf.constant_initializer(0.0))
            conv = tf.nn.conv2d(act, weights, strides=[
                1, 1, 1, 1], padding='SAME')
            net = tf.add(conv, bias, name='Add')
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('bias', bias)
        with tf.variable_scope('Elementwise_add'):
            net = tf.scalar_mul(multiplier, net)
            net = tf.add(x, net, name='Add')

    return net

# convolution layer


def Conv2d(x, n_filters_in, n_filters_out, name_scope='Conv_1', activiation=None):
    with tf.variable_scope(name_scope):
        weights = tf.get_variable("weights", shape=[3, 3, n_filters_in, n_filters_out],
                                  initializer=tf.truncated_normal_initializer(stddev=0.1))
        bias = tf.get_variable("bias", shape=[n_filters_out],
                               initializer=tf.constant_initializer(0.0))
        conv = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.add(conv, bias, name='Add')
        if activiation == 'relu':
            act = tf.nn.relu(net, name='relu')
            tf.summary.histogram('relu', act)
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
            act = tf.nn.relu(net, name='relu')    # NOTE: ACTIVATION IS REQUIRED HERE
            net = act
            tf.summary.histogram('weights', weights)
            tf.summary.histogram('bias', bias)
            tf.summary.histogram('relu', act)
        with tf.variable_scope('Subpixel_shuffle'):
            net = subpixel(net, factor=2)
    return net
