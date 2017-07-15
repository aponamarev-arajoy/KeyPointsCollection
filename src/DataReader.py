#!/usr/bin/env python
"""
DataReader is a utility for reading the data.
Created 7/12/17 at 6:36 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import tensorflow as tf
from tensorflow import TFRecordReader, parse_single_example, FixedLenFeature

def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH, num_classes=1, batch_size=5):
    reader = TFRecordReader()

    _, serialized_example = reader.read(filename_queue)

    features = parse_single_example(
        serialized_example,
        # Defaults are not specified since both keys are required.
        features={
            'image/height': FixedLenFeature([], tf.int64),
            'image/width': FixedLenFeature([], tf.int64),
            'image/encoded': FixedLenFeature([], tf.string),
            'seg/encoded': FixedLenFeature([], tf.string)
        })

    # Convert from a scalar string tensor (whose single string has
    image = tf.decode_raw(features['image/encoded'], tf.uint8)
    mask = tf.decode_raw(features['seg/encoded'], tf.uint8)

    height = tf.cast(features['image/height'], tf.int64)
    width = tf.cast(features['image/width'], tf.int64)

    #image_shape = tf.stack([height, width, 3])
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
    #mask_shape = tf.stack([height, width, 1])
    mask_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, num_classes), dtype=tf.int32)


    image = tf.reshape(image, image_size_const)
    mask = tf.reshape(mask, mask_size_const)

    # Random transformations can be put here: right before you crop images
    # to predefined size. To get more information look at the stackoverflow
    # question linked above.
    images, annotations = tf.train.shuffle_batch([image, mask],
                                                 batch_size=batch_size,
                                                 capacity=30,
                                                 num_threads=2,
                                                 min_after_dequeue=10)

    return images, annotations