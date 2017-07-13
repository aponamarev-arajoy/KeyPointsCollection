#!/usr/bin/env python
"""
TestTFRecord is utility that allows to visualze the data stored in a tfrecord.
Created 7/12/17 at 11:48 AM.

This script is based on tutorial http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


import tensorflow as tf
from tensorflow.python.lib.io.tf_record import tf_record_iterator
import numpy as np
from matplotlib import pyplot as plt




pathToTFRecord = "../dataset/field_segmentation.tfrecords"


record_iterator = tf_record_iterator(pathToTFRecord)


for strign_record in record_iterator:

    example = tf.train.Example()
    example.ParseFromString(strign_record)

    h = int(example.features.feature['image/height'].int64_list.value[0])
    w = int(example.features.feature['image/width'].int64_list.value[0])
    img_str = example.features.feature['image/encoded'].bytes_list.value[0]
    mask = example.features.feature['seg/encoded'].bytes_list.value[0]

    img_reconstructed = np.fromstring(img_str, np.uint8)
    img_reconstructed = np.reshape(img_reconstructed, (h,w,-1))
    plt.imshow(img_reconstructed)
    mask_reconstructed = np.fromstring(mask, np.uint8)
    mask_reconstructed = np.reshape(mask_reconstructed, (h,w))
    plt.imshow(mask_reconstructed, cmap="gray")