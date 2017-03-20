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

""" The base class for RNN modules """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

import tensorflow as tf
import tensorflow.contrib.framework as framework
import tensorflow.contrib.layers as layers


class RNN(object):
    """ Base class for RNN model classes """
    def __init__(self, scope=None):
        """ Create an RNN object """
        self.states = []
        self.outputs = []

        with tf.variable_scope(scope) as variable_scope:
            self.variable_scope = variable_scope

    @property
    def output(self):
        """ Get the output for the current step """
        return self.outputs[-1] if len(self.outputs) > 0 else None

    @property
    def state(self):
        """ Get the state for the current step """
        return self.states[-1] if len(self.states) > 0 else None

    @property
    def step(self):
        """ How many steps have been """
        return len(self.outputs)

    @framework.add_arg_scope
    def collect_named_outputs(self, tensor, outputs_collections=None):
        """ Wrapper for collect_named_outputs """
        alias = self.variable_scope.original_name_scope
        return layers.utils.collect_named_outputs(outputs_collections, alias, tensor)

    @staticmethod
    def step_fn(wrapped_fn):
        """ Wrap an RNN class method's step function to implement basic expected behavior """
        @functools.wraps(wrapped_fn)
        def wrapper(self, *args, **kwargs):
            """ Determine scope reuse and keep track of states and outputs """
            reuse = True if self.step > 0 else None
            with framework.arg_scope(
                [self.collect_named_outputs],
                outputs_collections=kwargs.get('outputs_collections')):

                with tf.variable_scope(self.variable_scope, reuse=reuse):
                    output, state = wrapped_fn(self, *args, **kwargs)
                    output = tf.identity(output, name='rnn_output')

                    self.outputs.append(output)
                    self.states.append(state)

                    return self.collect_named_outputs(output)

        return wrapper
