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

""" The Cell model """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import abc
from collections import namedtuple
from math import pi

import tensorflow as tf
import tensorflow.contrib.layers as layers

from glas.utils.ops import reduce_prod


EPSILON = 1e-3
ATTENTION_TYPES = [
    'none',
    'sigmoid',
    'softmax',
    'content',
    'grid',
    'gaussian',
    'cauchy'
]


class Attention(object):
    """ Base class for attention """
    __metaclass__ = abc.ABCMeta

    def __init__(self, epsilon=EPSILON, scope='Attention'):
        """ Initialize the attention """
        self.scope = scope
        self.epsilon = epsilon

    @abc.abstractmethod
    def read_size(self, data):
        """ Return the read size of the given data """
        raise NotImplementedError('Unknown read size')

    @abc.abstractmethod
    def read(self, data, focus):
        """ Do a read given the data """
        raise NotImplementedError('Reading is not supported')

    @abc.abstractmethod
    def read_multiple(self, data_list, focus):
        """ Do a filtered read for multiple tensors using the same focus """
        raise NotImplementedError('Reading is not supported')

    @abc.abstractmethod
    def write(self, data):
        """ Do a write given the data """
        raise NotImplementedError('Writing is not supported')


class NoAttention(Attention):
    """ Base class for attention """
    def __init__(self, output_size, scope='NoAttention'):
        """ Initialize the attention """
        super(NoAttention, self).__init__(scope=scope)

        self.output_size = output_size

    def read_size(self, data):
        """ Return the read size of the given data """
        return tuple(data.get_shape().as_list()[1:])

    def read(self, data, focus):
        """ Do a read given the data """
        return layers.flatten(data)

    def read_multiple(self, data_list, focus):
        """ Do a filtered read for multiple tensors using the same focus """
        return tf.concat(1, [layers.flatten(data) for data in data_list])

    def write(self, data):
        """ Do a write given the data """
        return layers.linear(data, self.output_size, scope='write')


class SimpleAttention(Attention):
    """ Attention filter based on element wise attention """
    def __init__(self, focus_fn, scope='SimpleAttention'):
        """ Initialize the attention """
        super(SimpleAttention, self).__init__(scope=scope)
        self.focus_fn = focus_fn

    def read_size(self, data):
        """ Return the read size of the given data """
        return (reduce_prod(data.get_shape().as_list()[1:]),)

    def read(self, data, focus):
        """ Do a read given the data """
        focus = layers.linear(focus, data.get_shape().as_list()[-1])
        focused = tf.expand_dims(self.focus_fn(focus, name='focus'), 1)

        return layers.flatten(focused * data)

    def read_multiple(self, data_list, focus):
        """ Do a filtered read for multiple tensors using the same focus """
        focus = layers.linear(focus, data_list[0].get_shape().as_list()[-1])
        focused = tf.expand_dims(self.focus_fn(focus, name='focus'), 1)

        focus_list = []
        for data in data_list:
            focus_list.append(layers.flatten(focused * data))

        return tf.concat(1, focus_list)

    def write(self, data):
        """ Do a filtered write given the data """
        raise NotImplementedError('Writing is not supported')


class ContentAttention(Attention):
    """ Attention filter based on content attention (similar to Neural Turing Machine) """
    def __init__(self, shape, scope='ContentAttention'):
        """ Initialize the attention """
        super(ContentAttention, self).__init__(scope=scope)
        self.shape = shape

    def _get_key(self, focus):
        """ Get the key for the data """
        beta = layers.linear(focus, 1)
        key = layers.linear(focus, self.shape[1])

        return beta, tf.expand_dims(tf.nn.l2_normalize(key, -1), -1)

    def _address(self, beta, key, data):
        """ Address the data with the given key """
        data = tf.nn.l2_normalize(data, -1)
        content_weight = tf.nn.softmax(beta * tf.squeeze(tf.batch_matmul(data, key)))

        return layers.flatten(tf.expand_dims(content_weight, -1) * data)

    def read_size(self, data):
        """ Return the read size of the given data """
        return (reduce_prod(data.get_shape().as_list()[1:]),)

    def read(self, data, focus):
        """ Do a read given the data """
        beta, key = self._get_key(focus)
        return self._address(beta, key, data)

    def read_multiple(self, data_list, focus):
        """ Do a filtered read for multiple tensors using the same focus """
        beta, key = self._get_key(focus)

        focus_list = []
        for data in data_list:
            focus_list.append(self._address(beta, key, data))

        return tf.concat(1, focus_list)

    def write(self, data):
        """ Do a filtered write given the data """
        raise NotImplementedError('Writing is not supported')


_Grid = namedtuple('Grid', ('i', 'j', 'size'))


def _create_grid(size, shape):
    """ Create the grid parameters based on the attention size """
    # Passed in shape is (y, x)
    scale = max(*shape) - 1.0 / max(size - 1, 1)

    grid_i = tf.reshape(tf.linspace(0.0, size - 1, size) - 0.5 * (size + 1.0), [1, size, 1]) * scale
    grid_j = tf.reshape(tf.linspace(0.0, size - 1, size) - 0.5 * (size + 1.0), [1, size, 1]) * scale

    return _Grid(grid_i, grid_j, (size, size))


class GridAttention(Attention):
    """ Attention based on grid filters """
    def __init__(self, shape, read_size, write_size, epsilon=EPSILON, scope='GridAttention'):
        """ Initialize the attention """
        super(GridAttention, self).__init__(epsilon=epsilon, scope=scope)
        # Filter matrices are shape: Fx = [i, x], Fy = [j, y]

        # Passed in shape is (y, x)
        self.shape = tuple(shape)
        self.data_x = tf.reshape(tf.linspace(0.0, shape[1] - 1, shape[1]), [1, 1, shape[1]])
        self.data_y = tf.reshape(tf.linspace(0.0, shape[0] - 1, shape[0]), [1, 1, shape[0]])
        self.data_scale = (0.5 * (shape[1] + 1.0), (0.5 * (shape[0] + 1.0)))

        # read_size/write_size is (i, j)
        self.read_grid = _create_grid(read_size, shape) if read_size else None
        self.write_grid = _create_grid(write_size, shape) if write_size else None

    def _get_center(self, grid, offsets, stride):
        """ Return the center tensor using the passed in offset """
        # The grid is of size [1, grid_size, 1]. The incoming stride & offsets are [batch_size, 1],
        # so reshape to [batch_size, 1, 1]
        stride = tf.expand_dims(stride, -1)
        offsets = tuple(tf.expand_dims(offset, -1) for offset in offsets)

        center_x = self.data_scale[0] * (offsets[0] + 1) + grid.i * stride
        center_y = self.data_scale[1] * (offsets[1] + 1) + grid.j * stride

        return center_x, center_y

    def _get_filter(self, data, grid, scope=None):
        """ Generate an attention filter """
        with tf.variable_scope(scope, 'filter', [data]):
            x_offset, y_offset, log_stride, scale, log_gamma = tf.split(
                1, 5, layers.linear(data, 5, scope='parameters'))

            center = self._get_center(grid, (x_offset, y_offset), tf.exp(log_stride))

            scale = tf.expand_dims(scale, -1)
            filter_x = scale * (self.data_x - center[0])
            filter_y = scale * (self.data_y - center[1])

            return filter_x, filter_y, tf.exp(log_gamma)

    def get_filter(self, data, grid, scope=None):
        """ Generate an attention filter """
        filter_x, filter_y, gamma = self._get_filter(data, grid, scope=scope)

        # Ensure the sum over 'a' of F[i, a] = 1
        filter_x /= tf.maximum(tf.reduce_sum(filter_x, -1, keep_dims=True), self.epsilon)
        filter_y /= tf.maximum(tf.reduce_sum(filter_y, -1, keep_dims=True), self.epsilon)

        return filter_x, filter_y, gamma

    def read_size(self, data):
        """ Return the read size of the given data """
        return (reduce_prod(self.read_grid.size),)

    def _read(self, data, filter_x, filter_y, gamma):
        """ Read using the given filter """
        data = tf.reshape(data, (-1,) + self.shape)
        filter_x_transpose = tf.transpose(filter_x, [0, 2, 1])
        patch = tf.batch_matmul(filter_y, tf.batch_matmul(data, filter_x_transpose))

        return gamma * layers.flatten(patch)

    def read(self, data, focus):
        """ Do a filtered read given the data """
        if not self.read_grid:
            raise ValueError('Reading is not supported')

        filter_x, filter_y, gamma = self.get_filter(focus, self.read_grid, scope='read/filter')
        return self._read(data, filter_x, filter_y, gamma)

    def read_multiple(self, data_list, focus):
        """ Do a filtered read for multiple tensors using the same focus """
        if not self.read_grid:
            raise ValueError('Reading is not supported')

        filter_x, filter_y, gamma = self.get_filter(focus, self.read_grid, scope='read/filter')

        patches = []
        for data in data_list:
            patches.append(self._read(data, filter_x, filter_y, gamma))

        return tf.concat(1, patches)

    def write(self, data):
        """ Do a filtered write given the data """
        if not self.write_grid:
            raise ValueError('Writing is not supported')

        filter_x, filter_y, gamma = self.get_filter(data, self.write_grid, scope='write/filter')

        filter_y_transpose = tf.transpose(filter_y, [0, 2, 1])
        window = layers.linear(data, reduce_prod(self.write_grid.size))
        window = tf.reshape(window, (-1, self.write_grid.size[1], self.write_grid.size[0]))
        patch = tf.batch_matmul(filter_y_transpose, tf.batch_matmul(window, filter_x))

        return tf.reciprocal(tf.maximum(gamma, self.epsilon)) * layers.flatten(patch)


class GaussianAttention(GridAttention):
    """ Attention based on gaussian filters """
    def __init__(self, shape, read_size, write_size, epsilon=EPSILON, scope='GaussianAttention'):
        """ Initialize the attention """
        super(GaussianAttention, self).__init__(
            shape, read_size, write_size=write_size, epsilon=epsilon, scope=scope)

    def _get_filter(self, data, grid, scope=None):
        """ Generate an attention filter """
        with tf.variable_scope(scope, 'filter', [data]):
            x_offset, y_offset, log_stride, log_variance, log_gamma = tf.split(
                1, 5, layers.linear(data, 5, scope='parameters'))

            center = self._get_center(grid, (x_offset, y_offset), tf.exp(log_stride))

            scale = 2.0 * tf.square(tf.exp(log_variance / 2.0))
            scale = tf.expand_dims(tf.maximum(scale, self.epsilon), -1)

            filter_x = tf.exp(-tf.square(self.data_x - center[0]) / scale)
            filter_y = tf.exp(-tf.square(self.data_y - center[1]) / scale)

            return filter_x, filter_y, tf.exp(log_gamma)


class CauchyAttention(GridAttention):
    """ Attention based on Cauchy filters """
    def __init__(self, shape, read_size, write_size, epsilon=EPSILON, scope='CauchyAttention'):
        """ Initialize the attention """
        super(CauchyAttention, self).__init__(
            shape, read_size, write_size=write_size, epsilon=epsilon, scope=scope)

    def _get_filter(self, data, grid, scope=None):
        """ Generate an attention filter """
        with tf.variable_scope(scope, 'filter', [data]):
            x_offset, y_offset, log_stride, log_scale, log_gamma = tf.split(
                1, 5, layers.linear(data, 5, scope='parameters'))

            center = self._get_center(grid, (x_offset, y_offset), tf.exp(log_stride))

            scale = tf.expand_dims(tf.maximum(tf.exp(log_scale), self.epsilon), -1)
            filter_x = 1 + tf.square((self.data_x - center[0]) / tf.maximum(scale, self.epsilon))
            filter_y = 1 + tf.square((self.data_y - center[1]) / tf.maximum(scale, self.epsilon))

            filter_x = tf.reciprocal(tf.maximum(pi * scale * filter_x, self.epsilon))
            filter_y = tf.reciprocal(tf.maximum(pi * scale * filter_y, self.epsilon))

            return filter_x, filter_y, tf.exp(log_gamma)


def create_attention(attention_type, data_shape, read_size=None, write_size=None):
    """ Create the appropriate attention based on the passed in config """
    if attention_type == 'none':
        return NoAttention(reduce_prod(data_shape))
    elif attention_type == 'sigmoid':
        return SimpleAttention(tf.sigmoid, scope='SigmoidAttention')
    elif attention_type == 'softmax':
        return SimpleAttention(tf.nn.softmax, scope='SoftmaxAttention')
    elif attention_type == 'content':
        return ContentAttention(tuple(data_shape[:2]))
    elif attention_type == 'gaussian':
        grid_attention_cls = GaussianAttention
    elif attention_type == 'cauchy':
        grid_attention_cls = CauchyAttention
    elif attention_type == 'grid':
        grid_attention_cls = GridAttention
    else:
        raise TypeError('Unknown attention type: "{0}"'.format(attention_type))

    return grid_attention_cls(tuple(data_shape[:2]), read_size, write_size)
