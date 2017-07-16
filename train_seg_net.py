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
from tensorflow.python.platform.app import flags
from matplotlib import pyplot as plt
import tensorflow as tf
import sys

FLAGS = flags.FLAGS
flags.DEFINE_bool("restore", False, "Do you want to restore a model? Default value is True.")
flags.DEFINE_float("learning_rate", 1e-4, "Provide a value for learning rate. Default value is 1e-4")
flags.DEFINE_float("width", 1.0, "Set the net width multiple. Default is 1.0. Type Float")
flags.DEFINE_integer("batch_size", 5, "Set the size of the mini-batch. Default value: 5. Type Float")
flags.DEFINE_string("log_dir", "logs", "Provide logging directory for recovering and storing model. Default value is logs")


# Define main drivers
tfrecords_path = "dataset/field_segmentation.tfrecords"
log_dir = FLAGS.log_dir
learning_rate = FLAGS.learning_rate
num_classes = 1
restore_model = FLAGS.restore

# Define data provider
sys.stdout.write("\r>> Initializing dataprovider")
sys.stdout.flush()
file_queue = tf.train.string_input_producer([tfrecords_path])
input_image, input_mask = read_and_decode(file_queue, IMAGE_HEIGHT=512, IMAGE_WIDTH=1024,
                                          num_classes=num_classes, batch_size=FLAGS.batch_size, color_dist=True)

# Define net
sys.stdout.write("\r>> Dataprovider was initialized successfully. Starting Net initialization.")
sys.stdout.flush()
net = model(input_image, num_classes, is_training=True, keep_prob=FLAGS.width,
            width_multiplier=FLAGS.width, scope="Mobile_UNet")

# Define losses
total_loss = IOU_calc_loss(tf.cast(input_mask, tf.float32), net)
slim.summary.scalar("IoU_Loss", total_loss)
summary_op = slim.summary.merge(tf.get_collection(tf.GraphKeys.SUMMARIES))

# Configure session
sys.stdout.write("\r>> Net was initialized successfully. Preparing computational session.")
sys.stdout.flush()
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True


optimizer = tf.train.AdamOptimizer(learning_rate)
train_op = slim.learning.create_train_op(total_loss, optimizer)

with tf.Session(config=config) as sess:

    if restore_model:
        sys.stdout.write("\r>> Restoring trainable variables.")
        sys.stdout.flush()
        variable_to_restore = slim.get_variables_to_restore()
        restorer = tf.train.Saver(variable_to_restore)
        latest_checkpoint = tf.train.latest_checkpoint(log_dir)
        restorer.restore(sess, latest_checkpoint)

    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)

    sys.stdout.write("\r>> The pipeline preparation was completed. Kicking-off training.")
    sys.stdout.flush()
    slim.learning.train(train_op, log_dir, number_of_steps=int(1e6), save_summaries_secs=60 * 5,
                        save_interval_secs=60 * 30, summary_op=summary_op)

    coord.request_stop()
    coord.join(threads)

