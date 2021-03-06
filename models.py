from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import building blocks
from layers import *


class EDSR(object):
    def __init__(self, n_filters=256, n_blocks=16, n_input_channel=1):
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_input_channel = n_input_channel

    def generator(self, input, reuse=None, name_scope='Generator'):

        with tf.variable_scope(name_scope) as scope:
            """ The EDSR super-resolution generator """
            if reuse:
                scope.reuse_variables()

            net = Conv2d(input, self.n_input_channel, self.n_filters,
                         name_scope='Conv_1', activiation='relu')
            net_before_resblock = net

            if self.n_blocks > 0:
                with tf.variable_scope('Resblocks') as scope:
                    for i in range(self.n_blocks):
                        net = ResBlock_EDSR(net, self.n_filters,
                                            name_scope='Resblock_' + str(i + 1))

            net = Conv2d(net, self.n_filters, self.n_filters,
                         name_scope='Conv_2', activiation='relu')
            net = tf.add(net_before_resblock, net, name='Elementwise_Add')

            net = Upsampler(net, self.n_filters, name_scope='Upsampler_1')
            net = Upsampler(net, self.n_filters / 4, name_scope='Upsampler_2')

            net = Conv2d(net, self.n_filters / 16, 1,
                         activiation='tanh', name_scope='Conv_nc')

            return net



def g_DownSampler(input, n_filters=128, name_scope='Downsampler'):

    with tf.variable_scope(name_scope) as scope:

        net = Conv2d(input, 1, n_filters, activiation='relu',
                     name_scope='Conv_1')
        net = ResBlock_EDSR(net, n_filters, name_scope='Resblock_1')
        net = Conv2d(net, n_filters, n_filters, activiation='relu',
                     stride=2, name_scope='Downsampler_1')
        net = ResBlock_EDSR(net, n_filters, name_scope='Resblock_2')
        net = Conv2d(net, n_filters, n_filters, activiation='relu',
                     stride=2, name_scope='Downsampler_2')
        net = Conv2d(net, n_filters, 1, activiation='tanh',
                     name_scope='Conv_2')

        return net


class Discriminator(object):
    def __init__(self, n_input_channel=1, is_train=True):
        self.n_input_channel = n_input_channel
        self.is_train = is_train    # For different behavior of batch normalization

    def d_SRGAN(self, input, reuse=None, name_scope='Discriminator'):

        with tf.variable_scope(name_scope) as scope:
            if reuse:
                scope.reuse_variables()

            net = Conv2d(input, self.n_input_channel, 64, activiation='lrelu',
                         name_scope='Conv_n64s1k3')

            net = Conv2d(net, 64, 64, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_1')
            net = Conv2d(net, 64, 128, stride=1, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_2')
            net = Conv2d(net, 128, 128, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_3')
            net = Conv2d(net, 128, 256, stride=1, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_4')
            net = Conv2d(net, 256, 256, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_5')
            net = Conv2d(net, 256, 512, stride=1, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_6')
            net = Conv2d(net, 512, 512, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_7')
            # net = tf.reshape(net, shape=[-1, 512 * 3 * 3], name='Flatten')
            net = tf.layers.flatten(net, name='Flatten')
            net = Dense(net, 1024, activiation='lrelu',
                        name_scope='Dense_n1024')
            logits = Dense(net, 1, activiation='sigmoid',
                           name_scope='Dense_nc')
            net = tf.nn.sigmoid(logits, name='sigmoid')

            return net, logits

    def d_SMALL(self, input, reuse=None, name_scope='Discriminator'):

        with tf.variable_scope(name_scope) as scope:
            if reuse:
                scope.reuse_variables()

            net = Conv2d(input, self.n_input_channel, 64, activiation='lrelu',
                         name_scope='Conv_1')

            net = Conv2d(net, 64, 64, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_2')
            net = Conv2d(net, 64, 128, stride=1, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_3')
            net = Conv2d(net, 128, 128, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_4')
            net = Conv2d(net, 128, 256, stride=1, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_5')
            net = Conv2d(net, 256, 256, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_6')
            net = Conv2d(net, 256, 256, stride=2, bn=True, is_train=self.is_train,
                         activiation='lrelu', name_scope='Conv_7')
            net = tf.layers.flatten(net, name='Flatten')
            net = Dense(net, 1024, activiation='lrelu', name_scope='Dense_1')
            logits = Dense(net, 1, activiation='sigmoid', name_scope='Dense_2')
            net = tf.nn.sigmoid(logits, name='sigmoid')

            return net, logits
