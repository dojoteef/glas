# Copyright 2017 The Nader Akoury. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

""" The script to convert MNIST data to tf.train.Examples """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import tensorflow.contrib.learn as learn
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import dtypes

import glas.data.encode as encode_utils
from glas.utils.ops import get_folds


IMAGE_SHAPE = [28, 28, 1]
_ITEMS_TO_DESCRIPTIONS = {'image': 'A [28, 28, 1] image representing an MNIST digit.'}


def _assert_dtype(images):
    """ Make sure the images are of the correct data type """
    dtype = dtypes.as_dtype(images.dtype).base_dtype
    if dtype not in (dtypes.uint8, dtypes.float32):
        raise TypeError('Invalid image dtype {0}, expected uint8 or float32'.format(dtype))

    return dtype


def _correct_images(images):
    """ Convert images to be correct """
    # From the MNIST website: "Pixels are organized row-wise. Pixel values are 0 to 255. 0 means
    # background (white), 255 means foreground (black)."
    # The dataset does not transform the image such that 255 is black, so do that here.
    dtype = _assert_dtype(images)
    max_val = 255 if dtype == dtypes.uint8 else 1.0
    return max_val - images


def convert(directory, validation_size, num_threads=1):
    """ Convert MNIST data to tf.train.Examples """
    datasets = learn.datasets.mnist.read_data_sets(
        directory, dtype=tf.uint8, reshape=False, validation_size=validation_size)

    for name in ['train', 'validation', 'test']:
        data = getattr(datasets, name)
        images = _correct_images(data.images)

        encode_utils.encode_images(
            encode_utils.PNGEncoder(dtype=tf.uint8),
            images, name, directory, num_threads=num_threads)


def dataset(directory, subset, num_folds, fold, holdout):
    """ Return the mnist dataset """
    decoder = slim.tfexample_decoder.TFExampleDecoder(
        {'image/encoded': tf.FixedLenFeature([], tf.string),
         'image/format': tf.FixedLenFeature([], tf.string)},
        {'image': slim.tfexample_decoder.Image(shape=IMAGE_SHAPE, channels=1)}
    )

    filenames = encode_utils.get_filenames(directory, subset)
    filenames = get_folds(filenames, num_folds, fold, holdout)

    return slim.dataset.Dataset(
        filenames, tf.TFRecordReader, decoder,
        encode_utils.num_examples(filenames), _ITEMS_TO_DESCRIPTIONS,
        data_shape=IMAGE_SHAPE)


def dataset_preloaded(directory, subset, num_folds, fold, holdout):
    """ Return the mnist dataset """
    datasets = learn.datasets.mnist.read_data_sets(directory)
    images = getattr(datasets, subset).images
    images = get_folds(images, num_folds, fold, holdout)

    return slim.dataset.Dataset(
        images, None, None, images.shape[0], _ITEMS_TO_DESCRIPTIONS,
        data_shape=IMAGE_SHAPE)


def reshape_images(inputs):
    """ Reshape the images """
    images = tf.reshape(inputs, [-1] + IMAGE_SHAPE)
    return _correct_images(images)


def _main(argv=None):  # pylint: disable=unused-argument
    """ The main entry point """
    convert(FLAGS.directory, FLAGS.validation_size, FLAGS.num_threads)


def _parse_commandline():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--directory',
        type=str,
        default='data/mnist',
        action='store',
        help='The directory where to store the MNIST data'
    )
    parser.add_argument(
        '--num_threads',
        type=str,
        default=1,
        action='store',
        help='How many threads to use to process the images'
    )
    parser.add_argument(
        '--validation_size',
        type=str,
        default=5000,
        action='store',
        help='How many training examples to reserve for validation'
    )

    return parser.parse_args()


if __name__ == '__main__':
    FLAGS = _parse_commandline()
    tf.app.run(main=_main)
