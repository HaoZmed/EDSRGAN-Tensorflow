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
def resnet_edsr(x):
    pass