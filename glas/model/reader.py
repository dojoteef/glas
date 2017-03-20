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

""" The GLAS Reader model """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.framework as framework
import tensorflow.contrib.losses as losses

from glas.model.attention import create_attention
from glas.model.cell import Cell
from glas.model.sample import create_sampler
import glas.model.rnn as rnn
import glas.utils.graph as graph_utils
from glas.utils.ops import exponential_decay


class GLAS(rnn.RNN):
    """ GLAS model """
    def __init__(self, encoder, decoder, sampler, attention, scope='GLAS'):
        """ Initialize the GLAS model """
        super(GLAS, self).__init__(scope=scope)

        self.encoder = encoder
        self.decoder = decoder
        self.sampler = sampler
        self.attention = attention

    @rnn.RNN.step_fn
    def __call__(self, data, outputs_collections=None):
        """ Execute the next time step of the GLAS model """
        # Get the current output and decoded output (or zeros if they do not exist yet)
        output = self.output if self.step > 0 else tf.zeros_like(data)
        decoded = self.decoder.output if self.step > 0 else self.decoder.zero_output_like(data)

        with framework.arg_scope(
            [self.encoder.next, self.decoder.next, self.sampler.next],
            outputs_collections=outputs_collections):

            # calculate the error between the output and the data
            data_error = data - tf.sigmoid(output)

            # seletively read from the data
            read_data = self.attention.read_multiple([data, data_error], decoded)

            # encode the data, error, and current decoded output
            encoded = self.encoder(tf.concat(1, [read_data, decoded]))

            # sample from the approximate posterior
            sample = self.sampler(encoded)

            # decode from the latent space
            decoded = self.decoder(sample)

            written = self.attention.write(decoded)

        return output + written, None


class Generator(rnn.RNN):
    """ GLAS model """
    def __init__(self, decoder, sampler, attention, scope='GLAS'):
        """ Initialize the GLAS model """
        super(Generator, self).__init__(scope=scope)

        self.decoder = decoder
        self.sampler = sampler
        self.attention = attention

    @rnn.RNN.step_fn
    def __call__(self, outputs_collections=None):
        """ Execute the next time step of the GLAS model """
        # Get the current output and decoded output (or zeros if they do not exist yet)
        output = self.output if self.step > 0 else 0.0
        decoded = self.decoder.output if self.step > 0 else tf.zeros((1, self.decoder.output_size))

        with framework.arg_scope(
            [self.decoder.next, self.sampler.random_sample],
            outputs_collections=outputs_collections):

            # sample from the approximate posterior
            sample = self.sampler.random_sample()

            # decode from the latent space
            decoded = self.decoder(sample)

            written = self.attention.write(decoded)

        return output + written, None


def calculate_reconstruction_loss(targets, outputs):
    """ Calculate the reconstruction loss given the inputs and targets """
    loss = tf.nn.sigmoid_cross_entropy_with_logits(outputs, targets)
    losses.add_loss(tf.reduce_mean(tf.reduce_sum(loss, 1), name='reconstruction_loss'))


def initialize_latent_weights(config, dataset):
    """ Initialize the latent loss weights """
    latent_weights = tf.minimum(1.0 - exponential_decay(
        config.batch_size, config.latent_weights_growth_step,
        1.0 - config.latent_weights, 1.0 - config.latent_weights_growth_rate,
        dataset, staircase=False), config.latent_weights_maximum, name='latent_weights')
    tf.add_to_collection(graph_utils.GraphKeys.TRAINING_PARAMETERS, latent_weights)

    return latent_weights


def create_model(config, inputs, dataset):
    """ Create the GLAS model """
    sampler = create_sampler(config)
    attention = create_attention(
        config.attention_type, dataset.data_shape,
        read_size=config.attention_read_size, write_size=config.attention_write_size)
    encoder = Cell(config.num_units, config.num_layers, scope='Encoder')
    decoder = Cell(config.num_units, config.num_layers, scope='Decoder')

    model = GLAS(encoder, decoder, sampler, attention)

    for step in xrange(config.num_steps):
        with tf.name_scope('step{0}'.format(step)):
            model(inputs, outputs_collections=graph_utils.GraphKeys.RNN_OUTPUTS)

    latent_weights = initialize_latent_weights(config, dataset)
    sampler.calculate_latent_loss(latent_weights)
    calculate_reconstruction_loss(inputs, model.output)
    return [tf.sigmoid(output) for output in model.outputs]


def create_generator(config, data_shape):
    """ Create the GLAS model """
    sampler = create_sampler(config)
    attention = create_attention(
        config.attention_type, data_shape,
        read_size=config.attention_read_size, write_size=config.attention_write_size)
    decoder = Cell(config.num_units, config.num_layers, scope='Decoder')

    model = Generator(decoder, sampler, attention)

    for step in xrange(config.num_steps):
        with tf.name_scope('step{0}'.format(step)):
            model(outputs_collections=graph_utils.GraphKeys.RNN_OUTPUTS)

    return [tf.sigmoid(output) for output in model.outputs]
