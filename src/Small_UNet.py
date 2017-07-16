#!/usr/bin/env python
"""
Small_UNet is method for defining a feature extractor.
Created 7/12/17 at 1:24 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import tensorflow as tf
import tensorflow.contrib.slim as slim

from .NetUtils import _depthwise_separable_conv, _deconv, _lateral_connection

def model(inputs, num_classes=1, is_training=True, keep_prob=0.5, width_multiplier=1, scope='MobileNet'):
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=tf.nn.relu,
                            normalizer_fn=None):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=None):

                inputs = tf.identity(tf.cast(inputs, tf.float32) / 255.0, name="normalized_input")

                c1 = slim.convolution2d(inputs, round(32 * width_multiplier), 3, stride=2, padding='SAME', scope='conv_1')

                c2 = _depthwise_separable_conv(c1, 64, width_multiplier, downsample=True, sc='conv_ds_2')
                c3 = _depthwise_separable_conv(c2, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                c4 = _depthwise_separable_conv(c3, 256, width_multiplier, downsample=True, sc='conv_ds_4')
                c5 = _depthwise_separable_conv(c4, 512, width_multiplier, downsample=True, sc='conv_ds_5')
                c6 = _depthwise_separable_conv(c5, 1024, width_multiplier, downsample=True, sc='conv_ds_6')
                c7 = _depthwise_separable_conv(c6, 1024, width_multiplier, downsample=True, sc='conv_ds_7')
                c7 = slim.dropout(c7, keep_prob)

                # Upsampling path
                up6 = _deconv(c7, 1024, width_multiplier, sc="up6")
                up6 = _lateral_connection(up6, c6, 1024, width_multiplier, "l6")
                up5 = _deconv(up6, 512, width_multiplier, sc="up5")
                up5 = _lateral_connection(up5, c5, 512, width_multiplier, "l5")
                up4 = _deconv(up5, 256, width_multiplier, sc="up4")
                up4 = _lateral_connection(up4, c4, 256, width_multiplier, "l4")
                up3 = _deconv(up4, 128, width_multiplier, sc="up3")
                up3 = _lateral_connection(up3, c3, 128, width_multiplier, "l3")
                up2 = _deconv(up3, 64, width_multiplier, sc="up2")
                up2 = _lateral_connection(up2, c2, 64, width_multiplier, "l2")
                up1 = _deconv(up2, 32, width_multiplier, sc="up1")
                up1 = _lateral_connection(up1, c1, num_classes, 1, "l1")
                up0 = _deconv(up1, 32, width_multiplier, sc="up0")
                end_point = _lateral_connection(up0, inputs, num_classes, 1, "l0")
                end_point = tf.sigmoid(end_point, name="class_confidence")
                tf.add_to_collection(end_points_collection, end_point)

                return end_point


