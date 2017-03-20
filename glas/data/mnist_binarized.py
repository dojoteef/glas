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

""" The script to load binarized MNIST data """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow.contrib.learn as learn
import tensorflow.contrib.slim as slim

from glas.data.mnist import IMAGE_SHAPE


_ITEMS_TO_DESCRIPTIONS = {'image': 'A [28, 28, 1] image representing a binarized MNIST digit.'}
_DOWNLOAD_URL = 'http://www.cs.toronto.edu/~larocheh/public/datasets/binarized_mnist/'
_SUBSET_TO_FILENAME = {'train': 'train', 'validate': 'valid', 'test': 'test'}


def _get_filename(subset):
    """ Get the filename of the particular subset """
    return 'binarized_mnist_{0}.amat'.format(_SUBSET_TO_FILENAME[subset])


def dataset(directory, subset):
    """ Return the mnist dataset """
    filename = _get_filename(subset)
    local_file = learn.datasets.base.maybe_download(filename, directory, _DOWNLOAD_URL + filename)
    with open(local_file, 'r') as data_file:
        images = np.array([[np.float32(i) for i in line.split()] for line in data_file.readlines()])

    return slim.dataset.Dataset(
        images, None, None, images.shape[0], _ITEMS_TO_DESCRIPTIONS,
        data_shape=IMAGE_SHAPE)
