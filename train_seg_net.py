#!/usr/bin/env python
"""
train_seg_net is a training pipeline.
Created 7/12/17 at 6:34 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"


from src.DataReader import read_and_decode
from src.Small_UNet import model
from src.LossUtils import IOU_calc_loss
from tensorflow.contrib import slim
import tensorflow as tf
import sys


# Define main drivers
tfrecords_path = "dataset/field_segmentation.tfrecords"
log_dir = "logs"
learning_rate = 1e-3
num_classes = 1

# Define data provider
sys.stdout.write("\r>> Initializing dataprovider")
sys.stdout.flush()
file_queue = tf.train.string_input_producer([tfrecords_path])
input_image, input_mask = read_and_decode(file_queue, IMAGE_HEIGHT=512, IMAGE_WIDTH=1024, num_classes=num_classes)

# Define net
sys.stdout.write("\r>> Dataprovider was initialized successfully. Starting Net initialization.")
sys.stdout.flush()
net = model(input_image, num_classes, is_training=True, width_multiplier=1, scope="Mobile_UNet")

# Define losses
total_loss = IOU_calc_loss(tf.cast(input_mask, tf.float32), net)
slim.summary.scalar("IoU_Loss", total_loss)
summary_op = slim.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

# The op for initializing the variables.
init_op = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())

# Configure session
sys.stdout.write("\r>> Net was initialized successfully. Preparing computational session.")
sys.stdout.flush()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True


optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = slim.learning.create_train_op(total_loss, optimizer)

with tf.Session(config=config) as sess:

    sess.run(init_op)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sys.stdout.write("\r>> The pipeline preparation was completed. Kicking-off training.")
    sys.stdout.flush()
    slim.learning.train(train_op, log_dir, number_of_steps=int(1e6), save_summaries_secs=60 * 5,
                        save_interval_secs=60 * 30, summary_op=summary_op)

    coord.request_stop()
    coord.join(threads)

