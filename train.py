
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function


import csv

import os

from PIL import Image
import numpy as np
from scipy.misc import imread
from scipy.misc import imsave
import scipy.stats as st
from timeit import default_timer as timer
from PreProcess import preprocess

import tensorflow as tf
from nets import resnet_v2
from nets.mobilenet import mobilenet_v2

from io import BytesIO

import net

slim = tf.contrib.slim
model_dir = "" #the path stored checkpoint

tf.flags.DEFINE_string(
    'master', '', 'The address of the TensorFlow master to use.')

tf.flags.DEFINE_string(
    'checkpoint_path_resnet_v2', model_dir + 'resnet_enhanced.ckpt', 'Path to checkpoint for resnet v2 network.')

tf.flags.DEFINE_string(
    'input_dir', '', 'images for training')

tf.flags.DEFINE_integer(
    'image_width', 299, 'Width of each input images.')

tf.flags.DEFINE_integer(
    'image_height', 299, 'Height of each input images.')

tf.flags.DEFINE_integer(
    'batch_size',62, 'How many images process at one time.')

FLAGS = tf.flags.FLAGS

def load_target_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'data.csv')) as f:
        return {row[0]: int(row[2]) for row in csv.reader(f)}


def load_raw_class(input_dir):
    """Loads target classes."""
    with tf.gfile.Open(os.path.join(input_dir, 'data.csv')) as f:
        return {row[0]: int(row[1]) for row in csv.reader(f)}


def load_images(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:image = imread(f, mode='RGB').astype(np.float) / 255.0
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """

    image_mix = np.zeros(batch_shape) #Image preprocessed
    image_raw = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    filelist = os.listdir(input_dir)
    for item in filelist:
        raw_path = input_dir + '/' + item
        if item.endswith('.png'):
            image_tmp = (imread(raw_path, mode='RGB').astype(np.float) / 255.0)*2.0-1.0
            image = preprocess(image_tmp)
        else:
            continue
        image_raw[idx, :, :, :] = image_tmp
        image_mix[idx, :, :, :] = image
        filenames.append(os.path.basename(item))
        idx += 1
        if idx == batch_size:
            yield filenames, image_mix, image_raw
            filenames = []
            image_mix = np.zeros(batch_shape)
            image_raw = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, image_mix, image_raw


def load_images2(input_dir, batch_shape):
    """Read png images from input directory in batches.

    Args:
      input_dir: input directory
      batch_shape: shape of minibatch array, i.e. [batch_size, height, width, 3]

    Yields:image = imread(f, mode='RGB').astype(np.float) / 255.0
      filenames: list file names without path of each image
        Lenght of this list could be less than batch_size, in this case only
        first few images of the result are elements of the minibatch.
      images: array with all images from this batch
    """

    image_raw = np.zeros(batch_shape)
    filenames = []
    idx = 0
    batch_size = batch_shape[0]
    filelist = os.listdir(input_dir)
    for item in filelist:
        raw_path = input_dir + '/' + item
        if item.endswith('.png'):
            image_tmp = (imread(raw_path, mode='RGB').astype(np.float) / 255.0) * 2.0 - 1.0
        else:
            continue
        image_raw[idx, :, :, :] = image_tmp
        filenames.append(os.path.basename(item))
        idx += 1
        if idx == batch_size:
            yield filenames, image_raw
            filenames = []
            image_raw = np.zeros(batch_shape)
            idx = 0
    if idx > 0:
        yield filenames, image_raw

batch_shape = [FLAGS.batch_size, FLAGS.image_height, FLAGS.image_width, 3]


def main(_):
    # Images for inception classifier are normalized to be in [-1, 1] interval,
    # eps is a difference between pixels so it should be in [0, 2] interval.
    # Renormalizing epsilon from [0, 255] to [0, 2].
    full_start = timer()

    tf.logging.set_verbosity(tf.logging.INFO)

    with tf.Graph().as_default():

        # Prepare graph
        x_input = tf.placeholder(tf.float32, shape=batch_shape)
        x_raw = tf.placeholder(tf.float32, shape=batch_shape)
        raw_class_input = tf.placeholder(tf.int32, shape=[FLAGS.batch_size])
        learning_rate = tf.placeholder(tf.float32)


        one_hot_raw_class = tf.one_hot(raw_class_input, num_classes)
        x_224 = tf.image.resize_images(x_input, [224, 224], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
        with slim.arg_scope(resnet_v2.resnet_arg_scope()):
            logits, end_points = resnet_v2.resnet_v2_50(x_224, 1001, is_training=True, scope='resnet_v2_50',
                                                        reuse=tf.AUTO_REUSE)
        loss = slim.losses.softmax_cross_entropy(end_points['predictions'], one_hot_raw_class)

        optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)
        s1 = tf.train.Saver(slim.get_model_variables(scope='resnet_v2_50'))
        init = tf.global_variables_initializer()
        logdir = "../saved_model"
        savepath = logdir + "/resnet_enhanced.ckpt"

        print('Created Graph')
        # Run computation
        with tf.Session() as sess:
            processed = 0.0
            sess.run(init)
            s1.restore(sess, FLAGS.checkpoint_path_resnet_v2)
            #warm up
            """
            for i in range(10):
                for filenames, image_raw in load_images2(FLAGS.input_dir, batch_shape):
                    raw_class_for_batch = (
                            [all_images_raw_class[n] for n in filenames]
                            + [0] * (FLAGS.batch_size - len(filenames)))
                    _, losses = sess.run([optimizer, loss],
                                         feed_dict={x_input: image_raw, raw_class_input: raw_class_for_batch,x_raw:image_raw})
                    print(losses)
            """
            epoch = 100
            #Augumented data, firstly set high learning rate then reduce slowly
            lr = 1e-3
            for i in range(epoch):
                for filenames, image_mix, image_raw in load_images(FLAGS.input_dir, batch_shape):
                    raw_class_for_batch = (
                        [all_images_raw_class[n] for n in filenames]
                        + [0] * (FLAGS.batch_size - len(filenames)))
                    _, losses = sess.run([optimizer, loss], feed_dict={x_input: image_mix, x_raw:image_raw,raw_class_input: raw_class_for_batch,learning_rate: lr})
                    print(losses)
                    if(losses>=6.7):
                        #print('GRADIENT DISVERAGED! RESTORE IMMEDIATELY! \nIF THIS SITUATION REPEATS, REDUCE THE LEARNING RATE!')
                        s1.restore(sess, FLAGS.checkpoint_path_resnet_v2)
                        continue
                    s1.save(sess,savepath)
                lr = lr * (0.9 - epoch * 0.1)
                lr = np.clip(lr, 1e-6, 1e-1)

            processed += FLAGS.batch_size
            full_end = timer()
            print("DONE: Processed {} images in {} sec".format(processed, full_end - full_start))


if __name__ == '__main__':
    all_images_raw_class = load_raw_class(FLAGS.input_dir)
    tf.app.run()

