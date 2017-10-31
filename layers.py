"""
Implementation of EDSR
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 

### subpixel shuffle
def subpixel(x, factor, name):

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
        list_x = tf.split(x, factor, axis=3)
        x = tf.concat(list_x, axis=2)
        output_channel = int(input_channel / factor**2)
        x = tf.reshape(x, [batch_size, height*factor, width*factor, output_channel])
        return x

    with tf.variable_scope(name) as scope:
        return _phase_shift(x, factor)


### resnet block - EDSR style
def ResBlock_EDSR(x, n_filters, multiplier=0.1, name_scope='resblock'):
    """
    Define a ResBlock specified by the EDSR paper. This is DIFFERENT
    from the conventional ResBlock
    """
    with tf.variable_scope(name_scope):

        with tf.name_scope('conv_1') as scope: 
            weights = tf.get_variable("conv_1/weights", shape=[3, 3, n_filters, n_filters],
                initializer=tf.truncated_normal_initializer)
            bias = tf.get_variable("conv_1/bias", shape=[n_filters],
                initializer=tf.constant_initializer(0.0))
            net = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
            act = tf.nn.relu(tf.add(net, bias), name='conv_1/activation_1')

            tf.summary.histogram('conv_1/weights', weights)
            tf.summary.histogram('conv_1/bias', bias)
            tf.summary.histogram('conv_1/activation', act)

        with tf.name_scope('conv_2') as scope:
            weights = tf.get_variable("conv_2/weights", shape=[3, 3, n_filters, n_filters],
                initializer=tf.truncated_normal_initializer)
            bias = tf.get_variable("conv_2/bias", shape=[n_filters],
                initializer=tf.constant_initializer(0.0))
            net = tf.nn.conv2d(act, weights, strides=[1, 1, 1, 1], padding='SAME')
            act = tf.nn.relu(tf.add(net, bias), name='conv_2/activation_2')

            tf.summary.histogram('conv_2/weights', weights)
            tf.summary.histogram('conv_2/bias', bias)
            tf.summary.histogram('conv_2/activation', act)

        with tf.name_scope('elementwise_add') as scope:
            net = tf.scalar_mul(multiplier, net)
            out = tf.add(x, net, name='conv_2/elementwise_addition')
    
    return out

### convolution layer
def Conv2d(x, n_filters_in, n_filters_out, name_scope='Conv'):

    with tf.variable_scope(name_scope) as scope:

        weights = tf.get_variable("weights", shape=[3, 3, n_filters_in, n_filters_out],
            initializer=tf.truncated_normal_initializer)
        bias = tf.get_variable("bias", shape=[n_filters_out],
            initializer=tf.constant_initializer(0.0))
        net = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
        act = tf.nn.relu(tf.add(net, bias), name='activation_2')

        tf.summary.histogram('weights', weights)
        tf.summary.histogram('bias', bias)
        tf.summary.histogram('activation', act)

        return act

### Upsampler
def Upsampler(x, name_scope='Upsampler'):

    with tf.variable_scope(name_scope):
        
        with tf.name_scope('Conv'):
            # GET SHAPE OF x
            n_filters = int(x.get_shape()[-1])
            weights = tf.get_variable("conv/weights", shape=[3, 3, n_filters, n_filters],
                initializer=tf.truncated_normal_initializer)
            bias = tf.get_variable("conv/bias", shape=[n_filters],
                initializer=tf.constant_initializer(0.0))
            net = tf.nn.conv2d(x, weights, strides=[1, 1, 1, 1], padding='SAME')
            act = tf.nn.relu(tf.add(net, bias), name='conv/activation_1')

            tf.summary.histogram('conv/weights', weights)
            tf.summary.histogram('conv/bias', bias)
            tf.summary.histogram('conv/activation', act)

        return subpixel(act, factor=2, name='subpixel')

        