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

""" The script to convert Omniglot data to tf.train.Examples """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import scipy.io
import tensorflow as tf
import tensorflow.contrib.learn as learn
import tensorflow.contrib.slim as slim
from tensorflow.python.framework import dtypes

from glas.utils.ops import get_folds


IMAGE_SHAPE = [28, 28, 1]
_ITEMS_TO_DESCRIPTIONS = {'image': 'A [28, 28, 1] image representing a binarized Omniglot digit.'}
_DOWNLOAD_URL = 'https://github.com/yburda/iwae/raw/master/datasets/OMNIGLOT/chardata.mat'
_SUBSET_TO_GROUP = {'train': 'data', 'validate': 'data', 'test': 'testdata'}
_VALIDATION_SIZE = 1345


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


def dataset(directory, subset, num_folds, fold, holdout):
    """ Return the mnist dataset """
    local_file = learn.datasets.base.maybe_download('omniglot.mat', directory, _DOWNLOAD_URL)
    data = scipy.io.loadmat(local_file)

    images = data[_SUBSET_TO_GROUP[subset]].astype(np.float32)
    images = images.transpose([1, 0]).reshape([-1] + IMAGE_SHAPE)

    if subset == 'train':
        images = images[:-_VALIDATION_SIZE]
    elif subset == 'validate':
        images = images[-_VALIDATION_SIZE:]

    images = get_folds(images, num_folds, fold, holdout)
    return slim.dataset.Dataset(
        images, None, None, images.shape[0], _ITEMS_TO_DESCRIPTIONS,
        data_shape=IMAGE_SHAPE)


def reshape_images(inputs):
    """ Reshape the images """
    images = tf.reshape(inputs, [-1] + IMAGE_SHAPE)
    return _correct_images(images)
