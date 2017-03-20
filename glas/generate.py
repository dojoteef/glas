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

""" Main entry point for generating data from the GLAS model. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse

import tensorflow as tf
import tensorflow.contrib.layers as layers
import tensorflow.contrib.training as training

import glas.data.inputs as input_utils
from glas.model.reader import create_generator
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
        '--count',
        type=int,
        default=64,
        action='store',
        help='How many elements to generate'
    )

    return parser.parse_known_args()


def evaluate_model(config):
    """ Train the model using the passed in config """
    ###########################################################
    # Generate the model
    ###########################################################
    outputs = create_generator(config, input_utils.get_data_shape(config.dataset))

    ###########################################################
    # Setup the evaluation metrics and summaries
    ###########################################################
    # Generate the canvases that lead to the final output image
    summaries = []
    summaries.extend(layers.summarize_collection(graph_utils.GraphKeys.RNN_OUTPUTS))
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
    if tf.gfile.IsDirectory(checkpoint_path):
        checkpoint_path = tf.train.latest_checkpoint(checkpoint_path)
    eval_ops = tf.group(*outputs)
    hooks = [
        training.SummaryAtEndHook(FLAGS.log_dir, summary_op),
        training.StopAfterNEvalsHook(FLAGS.count)]

    training.evaluate_once(checkpoint_path, hooks=hooks, eval_ops=eval_ops)


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
