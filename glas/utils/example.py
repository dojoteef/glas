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

""" Utility functions for tf.train.Examples """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


# According to the tensorflow/core/example/example.proto Features can only be one of three types:
# (1) BytesList, FloatList, or Int64List), so allow for the creation of each.
def int64_feature(value):
    """ Create an int64 feature from the passed in value """
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def bytes_feature(value):
    """ Create an bytes feature from the passed in value """
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[tf.compat.as_bytes(value)]))


def float_feature(value):
    """ Create an float feature from the passed in value """
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))
