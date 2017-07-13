#!/usr/bin/env python
"""
cvtToTFRecord is a utility scrip that coverts data in dataset folder into a tf record for image segmentation.
Created 7/11/17 at 9:18 PM.

Based on the following tutorial:
http://warmspringwinds.github.io/tensorflow/tf-slim/2016/12/21/tfrecords-guide/
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

from glob import glob
from xmltodict import parse
from cv2 import imread, cvtColor, COLOR_BGR2RGB, fillPoly, resize
from os.path import join
from tensorflow.python.lib.io.tf_record import TFRecordWriter
from src.dataset_utils import segnemtation_image_to_tfexample
import sys
import numpy as np
from collections import namedtuple
Shape = namedtuple("Shape", ["h", "w"])


path_to_xmls = "../dataset/annotations/*.xml"
path_to_imgs = "../dataset/images"
tfrecords_filename = "../dataset/field_segmentation.tfrecords"

im_shape = Shape(h=512, w=1024)




def parse_xml(file):
    with open(file) as f:
        tree = parse(f.read())

    file_name = tree['annotation']['filename']
    obj = tree['annotation']['object']
    obj = obj['polygon']
    vertices = obj['pt']

    return file_name, vertices

def read_rgbimg(file):
    im = imread(file)
    im = cvtColor(im, COLOR_BGR2RGB)
    return im

def cvt_vertices(labelme_vertices):
    return [(p['x'], p['y']) for p in labelme_vertices]

def create_mask(img, vertices):
    vertices = np.array(vertices, dtype=np.uint32)
    mask = np.zeros_like(img).astype(np.uint8)
    mask = fillPoly(mask, np.int_([vertices]), (1,0,0))

    return mask[:,:,0]



def main():

    # Initialize a TFRecordWriter
    writer = TFRecordWriter(tfrecords_filename)

    # 1. Collect all xml files
    xml_files = glob(path_to_xmls)

    for p in xml_files:
        try:
            # 2. Parse out file name and vertices from an xml
            file_name, vertices = parse_xml(p)
            # 3. Read an image
            im = read_rgbimg(join(path_to_imgs, file_name))
            # 4. Convert vertices and create mask
            vertices = cvt_vertices(vertices)
            mask = create_mask(im, vertices)

            # 5. Resize images into set size
            im = resize(im, (im_shape.w, im_shape.h))
            mask = resize(mask, (im_shape.w, im_shape.h))

            height = im.shape[0]
            width = im.shape[1]

            example = segnemtation_image_to_tfexample(im.tostring(), height, width, mask.tostring())

            sys.stdout.write("\r>> Converting image {}".format(p))
            sys.stdout.flush()

            writer.write(example.SerializeToString())
        except:
            print("Failed to process {}".format(p))

    writer.close()
    print("Conversion into tfrecords format is successfully completed.")

    return True

if __name__ == "__main__":
    main()