#!/usr/bin/env python
"""
convert_video is a pipeline intended to demonstrate the usecase of the a segmentation net on a video.
Created 7/13/17 at 3:32 PM.
"""
__author__ = "Alexander Ponamarev"
__email__ = "alex.ponamaryov@gmail.com"

from src.Small_UNet import model
from src.Visualization_Utils import visualize_segmentation
from tensorflow.python.platform.app import flags
from cv2 import resize
import tensorflow as tf
from tensorflow.contrib import slim
import sys

try:
    from moviepy.editor import VideoFileClip
except:
    import imageio
    imageio.plugins.ffmpeg.download()
    from moviepy.editor import VideoFileClip

IMAGE_HEIGHT=int(512/2)
IMAGE_WIDTH=int(1024/2)
num_classes=1
threshold = 0.75

FLAGS = flags.FLAGS
flags.DEFINE_bool("restore", True, "Do you want to restore a model? Default value is True.")
flags.DEFINE_string("log_dir", "logs", "Provide logging directory for recovering and storing model. Default value is logs")
flags.DEFINE_string("video", "video.mp4", "Provide a relative path to a video file to be processed. Default value: video.mpg")
flags.DEFINE_string("save_to", "output_video.mp4", "Provide a relative path for a resulting video. Default value: output_video.mp4")
flags.DEFINE_float("width", 1.0, "Set the net width multiple. Default is 1.0. Type Float")

computation = {"segmentation_layer": None, "input_img_placeholder": None, "session": None}

def process_frame(img):
    img= resize(img, (IMAGE_WIDTH, IMAGE_HEIGHT))
    segmented_image = visualize_segmentation(img,
                                             computation["segmentation_layer"],
                                             computation["input_img_placeholder"],
                                             computation["session"],
                                             threshold)
    return segmented_image

def main():
    input_image = tf.placeholder(tf.uint8, shape=[1, IMAGE_HEIGHT, IMAGE_WIDTH, 3], name="image_input")
    sys.stdout.write("\r>> Initializing UNet")
    sys.stdout.flush()

    net = model(input_image, num_classes, is_training=False, keep_prob=1.0,
                width_multiplier=FLAGS.width, scope="Mobile_UNet")

    # Configure session
    sys.stdout.write("\r>> Net was initialized successfully. Preparing computational session.")
    sys.stdout.flush()

    with tf.Session() as sess:

        seg_image = sess.graph.get_tensor_by_name("Mobile_UNet/class_confidence:0")
        computation["segmentation_layer"] = seg_image
        computation["input_img_placeholder"] = input_image
        computation["session"] = sess

        sys.stdout.write("\r>> Restoring trainable variables.")
        sys.stdout.flush()
        variable_to_restore = slim.get_variables_to_restore()
        restorer = tf.train.Saver(variable_to_restore)
        latest_checkpoint = tf.train.latest_checkpoint(FLAGS.log_dir)
        restorer.restore(sess, latest_checkpoint)

        sys.stdout.write("\r>> The computational session is fully initialized. Trainable variables were successfully restored.")
        sys.stdout.flush()

        clip = VideoFileClip(FLAGS.video)
        processed_clip = clip.fl_image(process_frame)
        processed_clip.write_videofile(FLAGS.save_to, audio=False)

    return True

if __name__ == "__main__":
    main()


