#!/usr/bin/env python
"""
NetUtils is a utility file that defines layers.
Created 7/12/17 at 1:45 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


import tensorflow as tf
from tensorflow.contrib.layers import xavier_initializer, relu
from tensorflow import concat
from tensorflow.contrib.slim import batch_norm, separable_convolution2d, conv2d_transpose, convolution2d
from tensorflow import variable_scope

def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier, sc='/depthwise_conv', downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    conv = separable_convolution2d(inputs, num_outputs=None, stride=_stride, depth_multiplier=1,
                                   kernel_size=3, scope=sc + '/depthwise_conv')
    conv = convolution2d(conv, num_pwc_filters, kernel_size=1, scope=sc + '/pointwise_conv')

    return conv

def _concat(*arg, axis=-1, name='merge'):
    return concat(*arg, axis=axis, name=name)

def _deconv(input, num_pwc_filters, width_multiplier=1, strides=2, padding="SAME", sc="/deconv"):
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    conv = conv2d_transpose(input, num_pwc_filters, kernel_size=3, stride=strides, padding=padding,scope=sc + "/deconv")
    return conv


def _lateral_connection(td, dt, num_pwc_filters, width_multiplier, sc):
    output = _concat((td, dt), name=sc+"/merge")
    return _depthwise_separable_conv(output, num_pwc_filters, width_multiplier, sc=sc+"/force_choice")