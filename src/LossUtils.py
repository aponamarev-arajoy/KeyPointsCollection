#!/usr/bin/env python
"""
LossUtils is a collection of loss functions implemented in Tensorflow/Slim.
Created 7/12/17 at 7:48 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

from tensorflow.contrib.slim import flatten
from tensorflow import reduce_sum, reduce_mean, identity, variable_scope
import tensorflow as tf


smooth = 1.

def IOU_calc(y_true, y_pred):
    with variable_scope("IoU"):
        y_true_f = flatten(y_true)
        y_pred_f = flatten(y_pred)
        intersection = reduce_sum(y_true_f * y_pred_f, axis=-1, name="intersection")
        union = identity(reduce_sum(y_true_f + y_pred_f, axis=-1)-intersection, name="union")

        return reduce_mean((intersection + smooth) / (union + smooth), name="IoU_smoothed")


def IOU_calc_loss(y_true, y_pred):
    with variable_scope("IoU_Loss"):
        total_loss = identity(1.0-IOU_calc(y_true, y_pred), name="loss")
        return total_loss