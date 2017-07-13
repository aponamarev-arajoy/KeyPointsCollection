#!/usr/bin/env python
"""
Small_UNet is method for defining a feature extractor.
Created 7/12/17 at 1:24 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import tensorflow as tf
import tensorflow.contrib.slim as slim

from .NetUtils import _depthwise_separable_conv, _upsample, _lateral_connection

def model(inputs, num_classes=1, is_training=True, keep_prob=0.5, width_multiplier=1, scope='MobileNet'):
    with tf.variable_scope(scope) as sc:
        end_points_collection = sc.name + '_end_points'
        with slim.arg_scope([slim.convolution2d, slim.separable_convolution2d],
                            activation_fn=None,
                            outputs_collections=[end_points_collection]):
            with slim.arg_scope([slim.batch_norm],
                                is_training=is_training,
                                activation_fn=tf.nn.relu):

                inputs = tf.identity(tf.cast(inputs, tf.float32) / 127.5 - 1.0, name="normalized_input")

                c1 = slim.convolution2d(inputs, round(32 * width_multiplier), [3, 3], padding='SAME', scope='conv_1')
                c1 = slim.batch_norm(c1, scope='conv_1/pw_batch_norm')

                c2  = _depthwise_separable_conv(c1, 64, width_multiplier, downsample=True, sc='conv_ds_2')
                c3 = _depthwise_separable_conv(c2, 128, width_multiplier, downsample=True, sc='conv_ds_3')
                c4 = _depthwise_separable_conv(c3, 256, width_multiplier, downsample=True, sc='conv_ds_4')
                c5 = _depthwise_separable_conv(c4, 512, width_multiplier, downsample=True, sc='conv_ds_5')
                c6 = _depthwise_separable_conv(c5, 1024, width_multiplier, downsample=True, sc='conv_ds_6')
                c7 = _depthwise_separable_conv(c6, 1024, width_multiplier, downsample=True, sc='conv_ds_7')
                c8 = _depthwise_separable_conv(c7, 1024, width_multiplier, downsample=True, sc='conv_ds_8')
                c8 = _depthwise_separable_conv(c8, 1024, width_multiplier, sc='conv_8')
                c8 = slim.dropout(c8, keep_prob)

                # Upsampling path

                up7 = _upsample(c8, 1024, width_multiplier, "up7")
                up7 = _lateral_connection(up7, c7, 1024, width_multiplier, "l7")
                up6 = _upsample(up7, 1024, width_multiplier, "up6")
                up6 = _lateral_connection(up6, c6, 1024, width_multiplier, "l6")
                up5 = _upsample(up6, 512, width_multiplier, "up5")
                up5 = _lateral_connection(up5, c5, 512, width_multiplier, "l5")
                up4 = _upsample(up5, 256, width_multiplier, "up4")
                up4 = _lateral_connection(up4, c4, 256, width_multiplier, "l4")
                up3 = _upsample(up4, 128, width_multiplier, "up3")
                up3 = _lateral_connection(up3, c3, 128, width_multiplier, "l3")
                up2 = _upsample(up3, 64, width_multiplier, "up2")
                up2 = _lateral_connection(up2, c2, 64, width_multiplier, "l2")
                up1 = _upsample(up2, 32, width_multiplier, "up1")
                end_point = _lateral_connection(up1, c1, num_classes, width_multiplier, "l1")
                end_point = tf.sigmoid(end_point, name="class_confidence")

                return end_point


