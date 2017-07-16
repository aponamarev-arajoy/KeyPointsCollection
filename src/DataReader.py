#!/usr/bin/env python
"""
DataReader is a utility for reading the data.
Created 7/12/17 at 6:36 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import tensorflow as tf
from tensorflow import TFRecordReader, parse_single_example, FixedLenFeature
from tensorflow.python.ops import random_ops, math_ops, control_flow_ops


def distort_color(image, color_ordering=0, fast_mode=True, scope=None):
  """Distort the color of a Tensor image.
  Each color distortion is non-commutative and thus ordering of the color ops
  matters. Ideally we would randomly permute the ordering of the color ops.
  Rather then adding that level of complication, we select a distinct ordering
  of color ops for each preprocessing thread.
  Args:
    image: 3-D Tensor containing single image in [0, 1].
    color_ordering: Python int, a type of distortion (valid values: 0-3).
    fast_mode: Avoids slower ops (random_hue and random_contrast)
    scope: Optional scope for name_scope.
  Returns:
    3-D Tensor color-distorted image on range [0, 1]
  Raises:
    ValueError: if color_ordering not in [0, 3]
  """
  with tf.name_scope(scope, 'distort_color', [image]):
    if fast_mode:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=48. / 255.)
        #image = tf.image.random_saturation(image, lower=0.2, upper=1.5)
      else:
        image = tf.image.random_saturation(image, lower=0.2, upper=1.5)
        #image = tf.image.random_brightness(image, max_delta=48. / 255.)
    else:
      if color_ordering == 0:
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
      elif color_ordering == 1:
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
      elif color_ordering == 2:
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
      elif color_ordering == 3:
        image = tf.image.random_hue(image, max_delta=0.2)
        image = tf.image.random_saturation(image, lower=0.5, upper=1.5)
        image = tf.image.random_contrast(image, lower=0.5, upper=1.5)
        image = tf.image.random_brightness(image, max_delta=32. / 255.)
      else:
        raise ValueError('color_ordering must be in [0, 3]')

    # The random_* ops do not necessarily clamp.
    return tf.clip_by_value(image, 0.0, 1.0)


def random_flip_left_right_segmentation(img, mask):

    uniform_random = random_ops.random_uniform([], 0, 1.0, seed=None)
    mirror_cond = math_ops.less(uniform_random, .5)

    img = control_flow_ops.cond(mirror_cond,
                                lambda: tf.image.flip_left_right(img),
                                lambda: img)
    mask = control_flow_ops.cond(mirror_cond,
                                 lambda: tf.image.flip_left_right(mask),
                                 lambda: mask)

    return img, mask


def read_and_decode(filename_queue, IMAGE_HEIGHT, IMAGE_WIDTH, num_classes=1, batch_size=5, color_dist=False):
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

    #height = tf.cast(features['image/height'], tf.int64)
    #width = tf.cast(features['image/width'], tf.int64)
    image_size_const = tf.constant((IMAGE_HEIGHT, IMAGE_WIDTH, 3), dtype=tf.int32)
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

    if color_dist:

        images = tf.identity(tf.cast(images, tf.float32) / 255.0, name="img_in_-1_1_range")

        def prep_data_augment(im):
            im = distort_color(im, color_ordering=0, fast_mode=True)
            return im

        images = tf.map_fn(prep_data_augment, images)
        images, annotations = tf.map_fn(lambda x: random_flip_left_right_segmentation(x[0], x[1]),
                                        (images, annotations), dtype=(tf.float32, tf.uint8))

        images = tf.identity(tf.cast(images * 255.0, tf.uint8), name="img_augmented")


    return images, annotations