from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np

# Import building blocks
from layers import *


class EDSR(object):
    def __init__(self, n_filters=64, n_blocks=16, n_input_channel=1):
        self.n_filters = n_filters
        self.n_blocks = n_blocks
        self.n_input_channel = n_input_channel

    def generator(self, input):
        """ The EDSR super-resolution generator """

        net = Conv2d(input, self.n_input_channel, self.n_filters,
            name_scope='conv_1', activiation='relu')
        net_before_resblock = net

        if self.n_blocks > 0:
            with tf.variable_scope('Resblocks') as scope:
                for i in range(self.n_blocks):
                    net = ResBlock_EDSR(net, self.n_filters,
                                        name_scope='Resblock_' + str(i + 1))

        net = Conv2d(net, self.n_filters, self.n_filters, name_scope='Conv_2', activiation='relu')
        net = tf.add(net_before_resblock, net, name='Elementwise_Add')

        net = Upsampler(net, self.n_filters, name_scope='Upsampler_1')
        net = Upsampler(net, self.n_filters / 4, name_scope='Upsampler_2')

        net = Conv2d(net, self.n_filters / 16, 1, name_scope='Conv_3')
        net = tf.nn.tanh(net, name='Activation_tanh')

        out = net
        return out
