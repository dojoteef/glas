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

""" Utilities for calculating Tensorflow statistics """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import re

from six import iteritems
import tensorflow as tf
import tensorflow.contrib.framework as framework
import tensorflow.contrib.losses as losses
from tensorflow.python.client import device_lib


# Reused scopes have an '_<digits>' for the reuse count which will count as a match when
# calling the scope_index function
_SCOPE_MATCH = re.compile(r'([a-zA-Z]+)(_\d+)?')
_INVALID_DEVICE_CHARACTERS = re.compile(r'[^\w]')
_CPU_OPERATIONS = set(('Variable', 'VariableV2', 'Placeholder'))


# Definition of a Tower
Tower = collections.namedtuple('Tower', ['device', 'outputs', 'scope'])


class GraphKeys(tf.GraphKeys):
    """ Extension to keys for tensorflow graph collections. """
    METRICS = 'metrics'
    RNN_OUTPUTS = 'rnn_outputs'
    TRAINING_PARAMETERS = 'training_parameters'


def get_available_devices(device_type=None):
    """ Get the number of available gpus """
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if not device_type or x.device_type == device_type]


def collect_devices(device_filters):
    """ Collect devices that match the passed in parameters """
    devices = []
    for device_type, count in iteritems(device_filters):
        device_list = get_available_devices(device_type=device_type)
        devices.extend(device_list[:count])

    return devices


def scope_index(full_scope, subscope):
    """ Get the index of the passed in scope from the parent scope """
    scopes = [_SCOPE_MATCH.match(scope).group(1) for scope in full_scope.split('/')]
    subscopes = subscope.split('/')
    count = len(subscopes)

    for index in (index for index, element in enumerate(scopes) if element == subscopes[0]):
        if scopes[index:index + count] == subscopes:
            return index

    return None


def device_fn(device):
    """ Returns a function that given a tf.Operation returns the device """
    def _get_device(operation):
        """ Given a tf.Operation get the device """
        if operation.type in _CPU_OPERATIONS:
            return '/cpu:0'
        else:
            return device(operation) if callable(device) else device

    return _get_device


def create_towers(model_fn, devices, *args, **kwargs):
    """ Create towers for the passed in devices """
    towers = []
    reuse = None
    with framework.arg_scope([framework.model_variable, framework.variable], device='/cpu:0'):
        for index, device in enumerate(devices):
            with tf.name_scope('tower{0}'.format(index)) as tower_scope:
                with tf.device(device_fn(device)):
                    with tf.variable_scope(tf.get_variable_scope(), reuse=reuse):
                        towers.append(Tower(device, model_fn(*args, **kwargs), tower_scope))
                        reuse = True

    return towers


def optimize_towers(optimizer, towers, clip_norm=None, **kwargs):
    """ Create towers for the passed in devices """
    all_tower_klds = []
    all_tower_losses = []
    all_tower_grads_and_vars = []

    num_towers = len(towers)
    regularization_losses = losses.get_regularization_losses()

    for tower in towers:
        with tf.device(tower.device):
            with tf.name_scope(tower.scope):
                # Scale based on number of towers
                tower_losses = losses.get_losses(tower.scope)
                total_tower_loss = tf.divide(tf.add_n(tower_losses), num_towers, name='total_loss')
                all_tower_losses.append(total_tower_loss)

                tower_klds = losses.get_losses('{0}.*kl_divergence'.format(tower.scope))
                if tower_klds:
                    total_tower_kld = tf.divide(tf.add_n(tower_klds), num_towers, name='total_kld')
                    all_tower_klds.append(total_tower_kld)

                if regularization_losses:
                    # Regularization losses are only calculated when the associated variable is
                    # created, not when it is reused, so only add the regularization losses once on
                    # the device of the first tower
                    regularization_loss = tf.add_n(regularization_losses, 'regularization_loss')
                    all_tower_losses.append(regularization_loss)
                    regularization_losses = None

                    total_tower_loss += regularization_loss

                grads_and_vars = optimizer.compute_gradients(total_tower_loss, **kwargs)
                if clip_norm:
                    grads_and_vars = [
                        (tf.clip_by_norm(gradients, clip_norm), variable)
                        for (gradients, variable) in grads_and_vars if gradients is not None]
                all_tower_grads_and_vars.append(grads_and_vars)

    grads_and_vars = []
    for grads_and_var in zip(*all_tower_grads_and_vars):
        # grads_and_var should be the gradients of each tower for the same variable
        variable = grads_and_var[0][1]
        gradients_list = [gradients for gradients, _ in grads_and_var if gradients is not None]

        if gradients_list:
            gradients = tf.add_n(gradients_list, name='{0}/gradient/sum'.format(variable.op.name))
            grads_and_vars.append((gradients, variable))

    return all_tower_klds, all_tower_losses, grads_and_vars
