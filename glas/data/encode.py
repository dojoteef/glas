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

""" Module for encoding images into instances of tf.train.Example """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob
import re
import os
import threading
import sys

from six.moves import xrange  # pylint: disable=redefined-builtin
import tensorflow as tf

import glas.utils.example as example_utils


_TFRECORDS_MATCH = re.compile(r'[^\W\d_]+(\d+)-(\d+)\.tfrecords')


def get_filenames(directory, subset):
    """ Get the list of filenames given the directory and subset name """
    return [filename for filename in glob.glob(os.path.join(
        directory, '{0}*.tfrecords'.format(subset)))]


def num_examples(filenames):
    """ Return the number of examples given the filenames """
    return reduce(lambda x, y: x + int(y[1]) - int(y[0]), [
        _TFRECORDS_MATCH.match(os.path.basename(filename)).groups()
        for filename in filenames], 0)


class PNGEncoder(object):
    """ Helper class to encode raw images into PNG format """
    def __init__(self, dtype=tf.uint8, compression=None):
        """ Initialize the PNGEncoder """
        self.session = tf.Session()

        self.raw_image = tf.placeholder(dtype=dtype)
        self.png_encode = tf.image.encode_png(self.raw_image, compression=compression)

    @property
    def format(self):
        """ Return the type of image format this encoder uses """
        return 'PNG'

    def encode(self, image_data):
        """ Encode the image as a PNG """
        return self.session.run(self.png_encode, feed_dict={self.raw_image: image_data})


def _thread_encode_images(thread_index, encoder, images, image_ranges, name, directory):
    """ Encode the passed in images """
    thread_ranges = image_ranges[thread_index]

    processed = 0
    image_height = images.shape[1]
    image_width = images.shape[2]
    image_channels = images.shape[3]
    for image_range in thread_ranges:
        filename = os.path.join(directory, '{0}{1}-{2}.tfrecords'.format(
            name, image_range[0], image_range[1]))

        with tf.python_io.TFRecordWriter(filename) as writer:
            for image_index in xrange(*image_range):
                encoded_image = encoder.encode(images[image_index])
                example = tf.train.Example(features=tf.train.Features(feature={
                    'image/height': example_utils.int64_feature(image_height),
                    'image/width': example_utils.int64_feature(image_width),
                    'image/channels': example_utils.int64_feature(image_channels),
                    'image/format': example_utils.bytes_feature(encoder.format),
                    'image/encoded': example_utils.bytes_feature(encoded_image),
                }))
                writer.write(example.SerializeToString())

                processed += 1
                if not processed % 1000:
                    print('[thread {0}]: processed {1} images'.format(thread_index, processed))
                    sys.stdout.flush()

    print('[thread {0}]: processed {1} images'.format(thread_index, processed))
    sys.stdout.flush()


def _partition(elements, num_groups):
    """ Partition the list into n nearly evenly sized lists """
    # See http://stackoverflow.com/a/2660034
    group_size = len(elements) / float(num_groups)

    def _slice(index):
        """ Return a slice """
        return slice(_round(index), _round(index + 1))

    def _round(index):
        """ Round to the nearest group size """
        return int(round(index * group_size))

    return [elements[_slice(index)] for index in xrange(num_groups)]


def encode_images(encoder, images, name, directory, images_per_file=10000, num_threads=1):
    """ Encode the passed in images as instances of tf.train.Example """
    # Generate ranges of images to encode per thread
    image_count = images.shape[0]
    num_files = image_count // images_per_file + 1

    image_ranges = [
        [index, min(index + images_per_file, image_count)]
        for index in xrange(0, num_files * images_per_file, images_per_file)
        if index < image_count]

    num_threads = min(num_threads, len(image_ranges))
    image_ranges = _partition(image_ranges, num_threads)

    print('Launching {0} encoder threads'.format(num_threads))
    sys.stdout.flush()

    threads = []
    coordinator = tf.train.Coordinator()
    for thread_index in xrange(len(image_ranges)):
        thread_args = (thread_index, encoder, images, image_ranges, name, directory)
        thread = threading.Thread(target=_thread_encode_images, args=thread_args)

        thread.start()
        threads.append(thread)

    coordinator.join(threads)
    print('Finished encoding images')
    sys.stdout.flush()
