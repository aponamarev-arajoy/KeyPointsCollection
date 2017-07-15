#!/usr/bin/env python
"""
Visualization_Utils is a set of scripts designed to help to visualize processed image frames.
Created 7/14/17 at 11:05 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

import numpy as np
from cv2 import addWeighted

def visualize_segmentation(img, seg_layer, img_input_ph, sess, threshold = 0.75):
    input_shape = img.shape
    assert len(input_shape) in [3,4], "Incorrect image rank. Expected image input should be of either hwc or bhwc format. The shape of the image provided is {}".format(input_shape)
    assert img.dtype == np.uint8, "Incorrect image data type. Expected data type is np.uint8. Provided datatype is {}".format(img.dtype.__name__)
    if len(input_shape) == 3:
        img = np.expand_dims(img, 0)
    pred_mask = sess.run(seg_layer, feed_dict={img_input_ph: img}).squeeze()
    pred_mask[pred_mask < threshold] = 0
    pred_mask = (pred_mask * 255).astype(np.uint8)
    empty_mask = np.zeros_like(pred_mask)
    pred_mask = np.array([pred_mask, empty_mask, empty_mask]).transpose(1, 2, 0)
    segmented_image = addWeighted(pred_mask, 0.8, img.squeeze(), 1, 0)
    return segmented_image