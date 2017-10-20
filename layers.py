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
    """
    batch_size, height, width, input_channel = x.get_shape().as_list()
    x = tf.reshape(x, (batch_size, height, width, factor, factor))
    list_x = tf.split(x, factor, axis=3)   # (batch_size, height, width, 1, factor) x factor
    x = tf.squeeze(tf.concat(list_x, axis=2))          # (batch_size, height, width x factor, factor)
    list_x = tf.split(x, factor, axis=3)   # (batch_size, height, width x factor, factor) 
    x = tf.concat(list_x, axis=1)   # (batch_size, height x factor, width x factor, 1) 
  
  input_channel = x.get_shape().as_list()[-1]
  output_channel = input_channel / factor**2

  if output_channel != 1:
    return x

  with tf.variable_scope(name) as scope:
    x = _phase_shift(x, factor)

  return x

def ResBlock(x, name_scope):
  """
  Define a ResBlock specified by the EDSR paper. This is DIFFERENT
  from the conventional ResBlock
  """
  multiplier = self.multiplier
  n_filters = self.n_filters
  with tf.variable_scope(name_scope) as scope:
    net = tf.layers.conv2d(x, n_filters, (3, 3), padding='same',
      activation=tf.nn.relu, name='conv1')
    net = tf.layers.conv2d(net, n_filters, (3, 3), padding='same',
      name='conv2')
    net = tf.scalar_mul(multiplier, net)
    out = tf.add(x, net, name='add')
  return out
