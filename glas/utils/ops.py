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

""" Additional utility operations """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.distributions as distributions
import tensorflow.contrib.framework as framework


def reduce_prod(iterable):
    """ Return the product of the iterable """
    return reduce(lambda x, y: x * y, iterable)


def exponential_decay(batch_size, num_epochs, initial_rate, decay_rate, dataset,
                      staircase=True, name=None):
    """ Get the exponential decay for the following parameters """
    global_step = framework.get_or_create_global_step()
    decay_steps = int(num_epochs * dataset.num_samples / batch_size)

    return tf.train.exponential_decay(
        initial_rate, global_step,
        decay_steps, decay_rate,
        staircase=staircase, name=name)


# This is modeled after the kullback_leibler.py setup in Tensorflow
_DISTANCES = {}


def hellinger(dist_a, dist_b, name=None):
    """ Compute the hellinger distance between two distributions.
    NOTE: The actual value being calculated is half the hellinger distance squared. """
    type_a = type(dist_a)
    type_b = type(dist_b)
    if type_a != type_b:
        raise NotImplementedError(
            'Hellinger distance is only allowed between the same types of distributions types!'
            'dist_a type: {0}, dist_b type: {1}'.format(type_a.__name__, type_b.__name__))
    hellinger_fn = _DISTANCES[type_a]
    if hellinger_fn is None:
        raise NotImplementedError('H2({0}, {0}) not registered!'.format(type_a.__name__))

    with tf.name_scope("Hellinger"):
        return hellinger_fn(dist_a, dist_b, name=name)


def register_hellinger(dist_cls):
    """ Decorator to register a Hellinger distance implementation function. """

    def wrapper(hellinger_fn):
        """ Perform the KL registration. """
        if not callable(hellinger_fn):
            raise TypeError('hellinger_fn {0} is not callable'.format(str(hellinger_fn)))
        if dist_cls in _DISTANCES:
            raise ValueError('H2({0}, {0}) has already been registered to: {1}'.format(
                dist_cls.__name__, _DISTANCES[dist_cls]))
        _DISTANCES[dist_cls] = hellinger_fn
        return hellinger_fn

    return wrapper


@register_hellinger(distributions.Normal)
def _hellinger_normal(dist_a, dist_b, name=None):
    """ Compute the Hellinger distance between two normal distributions """
    with tf.name_scope(name, 'hellinger_normal', [dist_a.mu, dist_b.mu]):
        # Add an epsilon to avoid divide by zero
        denominator = tf.maximum(tf.square(dist_a.sigma) + tf.square(dist_b.sigma), 1e-8)

        mu_op = -tf.divide(tf.square(dist_a.mu - dist_b.mu), 4.0 * denominator)
        sigma_op = tf.maximum(tf.divide(2.0 * dist_a.sigma * dist_b.sigma, denominator), 0.0)

        return 1.0 - tf.sqrt(sigma_op) * tf.exp(mu_op)
