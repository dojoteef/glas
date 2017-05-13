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

""" Main entry point for evaluating the GLAS model. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import math
import sys

import tensorflow as tf
import tensorflow.contrib.metrics as metrics
import tensorflow.contrib.layers as layers
import tensorflow.contrib.training as training

import glas.data.inputs as input_utils
from glas.model.reader import create_model
import glas.utils.config as config_utils
import glas.utils.graph as graph_utils
import glas.utils.image as image_utils


def parse_commandline():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument(
        '--checkpoint_path',
        type=str,
        default='',
        action='store',
        help='Where to load checkpoints from'
    )
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
        '--log_dir',
        type=str,
        default='eval',
        action='store',
        help='The log directory.'
    )
    parser.add_argument(
        '--loop',
        dest='once',
        action='store_false',
        help='Continuous evaluate'
    )
    parser.add_argument(
        '--once',
        dest='once',
        action='store_true',
        help='Evaluate only once.'
    )
    parser.set_defaults(once=False)

    return parser.parse_known_args()


def evaluate_model(config):
    """ Train the model using the passed in config """
    ###########################################################
    # Create the input pipeline
    ###########################################################
    with tf.name_scope('input_pipeline'):
        dataset = input_utils.get_dataset(
            config.datadir, config.dataset, config.datasubset,
            num_folds=config.fold_count, fold=config.fold, holdout=True)

        init_op, init_feed_dict, image = input_utils.get_data(
            config.dataset, dataset, config.batch_size,
            num_epochs=config.num_epochs,
            num_readers=config.num_readers)

        images = tf.train.batch(
            [image], config.batch_size,
            num_threads=config.num_preprocessing_threads,
            capacity=5 * config.batch_size)

    ###########################################################
    # Generate the model
    ###########################################################
    outputs = create_model(config, images, dataset)

    ###########################################################
    # Setup the evaluation metrics and summaries
    ###########################################################
    summaries = []
    metrics_map = {}
    for loss in tf.losses.get_losses():
        metrics_map[loss.op.name] = metrics.streaming_mean(loss)

    for metric in tf.get_collection(graph_utils.GraphKeys.METRICS):
        metrics_map[metric.op.name] = metrics.streaming_mean(metric)

    total_loss = tf.losses.get_total_loss()
    metrics_map[total_loss.op.name] = metrics.streaming_mean(total_loss)
    names_to_values, names_to_updates = metrics.aggregate_metric_map(metrics_map)

    # Create summaries of the metrics and print them to the screen
    for name, value in names_to_values.iteritems():
        summary = tf.summary.scalar(name, value, collections=[])
        summaries.append(tf.Print(summary, [value], name))

    summaries.extend(layers.summarize_collection(tf.GraphKeys.MODEL_VARIABLES))
    summaries.extend(layers.summarize_collection(graph_utils.GraphKeys.METRICS))
    summaries.extend(layers.summarize_collection(graph_utils.GraphKeys.RNN_OUTPUTS))
    summaries.extend(layers.summarize_collection(graph_utils.GraphKeys.TRAINING_PARAMETERS))

    images = input_utils.reshape_images(images, config.dataset)
    tiled_images = image_utils.tile_images(images)
    summaries.append(tf.summary.image('input_batch', tiled_images))

    # Generate the canvases that lead to the final output image
    with tf.name_scope('canvases'):
        for step, canvas in enumerate(outputs):
            canvas = input_utils.reshape_images(canvas, config.dataset)
            tiled_images = image_utils.tile_images(canvas)
            summaries.append(tf.summary.image('step{0}'.format(step), tiled_images))

    summary_op = tf.summary.merge(summaries, name='summaries')

    ###########################################################
    # Begin evaluation
    ###########################################################
    checkpoint_path = FLAGS.checkpoint_path
    eval_ops = tf.group(*names_to_updates.values())
    hooks = [
        training.SummaryAtEndHook(log_dir=FLAGS.log_dir, summary_op=summary_op),
        training.StopAfterNEvalsHook(math.ceil(dataset.num_samples / float(config.batch_size)))]

    eval_kwargs = {}
    eval_fn = training.evaluate_repeatedly
    if FLAGS.once:
        if tf.gfile.IsDirectory(checkpoint_path):
            checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
        eval_fn = training.evaluate_once
    else:
        assert tf.gfile.IsDirectory(checkpoint_path), (
            'checkpoint path must be a directory when using loop evaluation')

    eval_fn(checkpoint_path, hooks=hooks, eval_ops=eval_ops, **eval_kwargs)


def main(argv=None):  # pylint: disable=unused-argument
    """ The main entry point """
    config = config_utils.load(
        OVERRIDES,
        config_file=FLAGS.config_file,
        defaults_file=FLAGS.defaults_file)

    evaluate_model(config['glas'])


if __name__ == '__main__':
    FLAGS, OVERRIDES = parse_commandline()
    tf.logging.set_verbosity(tf.logging.__dict__[FLAGS.log_level])
    tf.app.run()
