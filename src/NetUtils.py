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

def _depthwise_separable_conv(inputs, num_pwc_filters, width_multiplier, sc, downsample=False):
    """ Helper function to build the depth-wise separable convolution layer.
    """
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    _stride = 2 if downsample else 1

    # skip pointwise by setting num_outputs=None
    depthwise_conv = separable_convolution2d(inputs, num_outputs=None, stride=_stride, depth_multiplier=1,
                                             kernel_size=[3, 3], scope=sc+'/depthwise_conv')

    bn = batch_norm(depthwise_conv, scope=sc+'/dw_batch_norm')
    pointwise_conv = convolution2d(bn, num_pwc_filters, kernel_size=[1, 1], scope=sc+'/pointwise_conv')
    bn = batch_norm(pointwise_conv, scope=sc+'/pw_batch_norm')

    return bn

def _concat(*arg, axis=3, name='merge'):
    return concat(*arg, axis=axis, name=name)

def _deconv(input, num_pwc_filters, width_multiplier, strides=[2, 2], padding="SAME", sc="deconv"):
    num_pwc_filters = round(num_pwc_filters * width_multiplier)
    conv = conv2d_transpose(input, num_pwc_filters, kernel_size=3, stride=strides, padding=padding, scope=sc + "/deconv")
    conv = batch_norm(conv, scope=sc + "/bn")

    return conv


def _upsample(input, num_pwc_filters, width_multiplier, scope):
    """
    Upsamples an input and adds another convolution layer
    :param input:
    :param num_pwc_filters:
    :param width_multiplier:
    :param name:
    :return: upsampled input with a given number of layers
    """
    d = _deconv(input, num_pwc_filters, width_multiplier, strides=2, padding='SAME', sc=scope + "/upsample")
    d = _depthwise_separable_conv(d, num_pwc_filters, width_multiplier, sc=scope+'/dw_conv')

    return d


def _lateral_connection(td, dt, num_pwc_filters, width_multiplier, sc):
    l = _depthwise_separable_conv(dt, num_pwc_filters, width_multiplier, sc=sc+"/L")
    output = concat((td, l), axis=-1, name=sc+"/concat")
    return _depthwise_separable_conv(output, num_pwc_filters, width_multiplier, sc=sc+"/force_choice")