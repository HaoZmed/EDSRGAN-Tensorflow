from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import numpy as np 

# Import building blocks
from layers import *


class EDSR(object):
    def __init__(self, n_filters=256, n_blocks=16):
        self.n_filters = n_filters
        self.n_blocks = n_blocks
    
    def generator(self, input):
        """ The EDSR super-resolution generator """

        net = Conv2d(input, 1, self.n_filters, name_scope='Conv1')

        with tf.variable_scope('Resblocks') as scope:
            for i in range(self.n_blocks):
                net = ResBlock_EDSR(net, self.n_filters,
                    name_scope='Resblock_'+str(i+1))

        # net = subpixel(net, factor=2, name='subpixel_1')
        # out = subpixel(net, factor=2, name='subpixel_2')
        net = Upsampler(net, name_scope='Upsampler_1')
        net = Upsampler(net, name_scope='Upsampler_2')
        
        out = Conv2d(net, 16, 1, name_scope='Conv_2')

        return out
