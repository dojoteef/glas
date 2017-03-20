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

""" Main entry point for training the GLAS model. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import tensorflow.contrib.framework as framework
import tensorflow.contrib.layers as layers
import tensorflow.contrib.losses as losses
from tensorflow.contrib.slim import learning

import glas.data.inputs as input_utils
from glas.model.reader import create_model
import glas.utils.config as config_utils
import glas.utils.graph as graph_utils
import glas.utils.image as image_utils
from glas.utils.ops import exponential_decay


def parse_commandline():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--config_file',
        type=str,
        default='',
        action='store',
        help='JSON config file for the arguments'
    )
    parser.add_argument(
        '--defaults_file',
        type=str,
        default='',
        action='store',
        help='JSON config file for the argument defaults'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='WARN',
        choices=['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        action='store',
        help='JSON config file for the argument defaults'
    )
    parser.add_argument(
        '--log_device_placement',
        default=False,
        action='store_true',
        help='Whether to log the device placement to stdout'
    )
    parser.add_argument(
        '--num_gpus',
        type=int,
        default=len(graph_utils.get_available_devices('GPU')),
        action='store',
        help='How many gpus to use for training.'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='train',
        action='store',
        help='The log directory.'
    )

    return parser.parse_known_args()


def create_training_model(config, inputs_queue, data_shape):
    """ Wrapper for create_model that takes an inputs queue """
    return create_model(config, inputs_queue.dequeue(), data_shape)


def train_model(config):
    """ Train the model using the passed in config """
    training_devices = [
        graph_utils.device_fn(device)
        for device in graph_utils.collect_devices({'GPU': FLAGS.num_gpus})]
    assert training_devices, 'Found no training devices!'

    ###########################################################
    # Create the input pipeline
    ###########################################################
    with tf.device('/cpu:0'), tf.name_scope('input_pipeline'):
        dataset = input_utils.get_dataset(
            config.datadir, config.dataset, 'train')

        init_op, init_feed_dict, image = input_utils.get_data(
            config.dataset, dataset, config.batch_size,
            num_epochs=config.num_epochs,
            num_readers=config.num_readers)

        inputs_queue = input_utils.batch_images(
            image, config.batch_size,
            num_threads=config.num_preprocessing_threads,
            num_devices=len(training_devices))

    ###########################################################
    # Generate the model
    ###########################################################
    towers = graph_utils.create_towers(
        create_training_model, training_devices, config, inputs_queue, dataset)
    assert towers, 'No training towers were created!'

    ###########################################################
    # Setup the training objectives
    ###########################################################
    with tf.name_scope('training'):
        with tf.device('/cpu:0'):
            learning_rate_decay_step = config.learning_rate_decay_step / len(towers)
            learning_rate = tf.maximum(
                exponential_decay(
                    config.batch_size, learning_rate_decay_step,
                    config.learning_rate, config.learning_rate_decay, dataset),
                config.learning_rate_min, name='learning_rate')
            tf.add_to_collection(graph_utils.GraphKeys.TRAINING_PARAMETERS, learning_rate)

            optimizer = tf.train.AdamOptimizer(learning_rate)

        # Calculate gradients and total loss
        tower_klds, tower_losses, grads_and_vars = graph_utils.optimize_towers(
            optimizer, towers, clip_norm=config.clip)
        total_kld = tf.add_n(tower_klds, name='total_kld') if tower_klds else None
        total_loss = tf.add_n(tower_losses, name='total_loss')

        # Gather update ops from the first tower (for updating batch_norm for example)
        global_step = framework.get_or_create_global_step()
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS, towers[0].scope)
        update_ops.append(optimizer.apply_gradients(grads_and_vars, global_step=global_step))

        update_op = tf.group(*update_ops)
        with tf.control_dependencies([update_op]):
            train_op = tf.identity(total_loss, name='train_op')

    ###########################################################
    # Collect summaries
    ###########################################################
    with tf.device('/cpu:0'):
        summaries = []
        summaries.extend(learning.add_gradients_summaries(grads_and_vars))
        summaries.extend(layers.summarize_collection(tf.GraphKeys.MODEL_VARIABLES))
        summaries.extend(layers.summarize_collection(graph_utils.GraphKeys.METRICS))
        summaries.extend(layers.summarize_collection(graph_utils.GraphKeys.RNN_OUTPUTS))
        summaries.extend(layers.summarize_collection(graph_utils.GraphKeys.TRAINING_PARAMETERS))

        images = input_utils.reshape_images(inputs_queue.dequeue(), config.dataset)
        tiled_images = image_utils.tile_images(images)
        summaries.append(tf.summary.image('input_batch', tiled_images))

        # Generate the canvases that lead to the final output image
        with tf.name_scope('canvases'):
            for step, canvas in enumerate(towers[0].outputs):
                canvas = input_utils.reshape_images(canvas, config.dataset)
                tiled_images = image_utils.tile_images(canvas)
                summaries.append(tf.summary.image('step{0}'.format(step), tiled_images))

        with tf.name_scope('losses'):
            if total_kld is not None:
                summaries.append(tf.summary.scalar('total_kld', total_kld))
            summaries.append(tf.summary.scalar('total_loss', total_loss))

            for loss in tower_losses:
                summaries.append(tf.summary.scalar(loss.op.name, loss))

            for loss in losses.get_losses():
                summaries.append(tf.summary.scalar(loss.op.name, loss))

        summary_op = tf.summary.merge(summaries, name='summaries')

    ###########################################################
    # Begin training
    ###########################################################
    init_op = tf.group(tf.global_variables_initializer(), init_op)
    session_config = tf.ConfigProto(
        allow_soft_placement=False,
        log_device_placement=FLAGS.log_device_placement)

    prefetch_queue_buffer = 2 * len(training_devices)
    number_of_steps = int(int(dataset.num_samples / config.batch_size) / len(training_devices))
    number_of_steps = number_of_steps * config.num_epochs - prefetch_queue_buffer

    tf.logging.info('Running %s steps', number_of_steps)
    learning.train(
        train_op, FLAGS.log_dir, session_config=session_config,
        global_step=global_step, number_of_steps=number_of_steps,
        init_op=init_op, init_feed_dict=init_feed_dict,
        save_interval_secs=config.checkpoint_frequency,
        summary_op=summary_op, save_summaries_secs=config.summary_frequency,
        trace_every_n_steps=config.trace_frequency if config.trace_frequency > 0 else None)


def main(argv=None):  # pylint: disable=unused-argument
    """ The main entry point """
    config = config_utils.load(
        OVERRIDES,
        config_file=FLAGS.config_file,
        defaults_file=FLAGS.defaults_file)

    train_model(config['glas'])


if __name__ == '__main__':
    FLAGS, OVERRIDES = parse_commandline()
    tf.logging.set_verbosity(tf.logging.__dict__[FLAGS.log_level])
    tf.app.run()
