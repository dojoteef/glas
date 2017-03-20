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

"""Main entry point for training/testing of the stereotyping model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import re

import six

from glas.data.inputs import DATASETS
import glas.model.attention as attention
import glas.model.sample as sample


_FLAG_REGEX = re.compile(r'--(?:no-)?([\w-]+)')


def _json_parser():
    """ Parse command line arguments. """
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument(
        '--attention_type',
        type=str,
        default='none',
        choices=attention.ATTENTION_TYPES,
        action='store',
        help='The type of attention to create.'
    )
    parser.add_argument(
        '--attention_read_size',
        type=int,
        default=2,
        action='store',
        help='The read size of the attention filter.'
    )
    parser.add_argument(
        '--attention_write_size',
        type=int,
        default=5,
        action='store',
        help='The write size of the attention filter.'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=64,
        action='store',
        help='How many images per batches.'
    )
    parser.add_argument(
        '--checkpoint_frequency',
        type=int,
        default=300,
        action='store',
        help='The number of seceonds between saving model checkpoints.'
    )
    parser.add_argument(
        '--clip',
        type=float,
        default=0,
        action='store',
        help='Clip gradients using the passed in value.'
    )
    parser.add_argument(
        '--datadir',
        type=str,
        default='data',
        action='store',
        help='Where to store the data.'
    )
    parser.add_argument(
        '--dataset',
        type=str,
        default='mnist',
        choices=DATASETS,
        action='store',
        help='Which dataset to use.'
    )
    parser.add_argument(
        '--datasubset',
        type=str,
        default='train',
        choices=['train', 'validate', 'test'],
        action='store',
        help='Which subset of the dataset to use.'
    )
    parser.add_argument(
        '--latent_read_size',
        type=int,
        default=2,
        action='store',
        help='How much to read from the latent space.'
    )
    parser.add_argument(
        '--latent_size',
        type=int,
        default=2,
        action='store',
        help='The size of the latent space.'
    )
    parser.add_argument(
        '--latent_weights',
        type=float,
        default=1.0,
        action='store',
        help='Mulitplier to scale the latent loss'
    )
    parser.add_argument(
        '--latent_weights_growth_rate',
        type=float,
        default=0.0,
        action='store',
        help='The rate at which the weights of the latent loss grows'
    )
    parser.add_argument(
        '--latent_weights_growth_step',
        type=float,
        default=25,
        action='store',
        help='How quickly the weights of the latent loss grows'
    )
    parser.add_argument(
        '--latent_weights_maximum',
        type=float,
        default=1.0,
        action='store',
        help='The maximum multiplier for scaling the latent loss'
    )
    parser.add_argument(
        '--learning_rate',
        type=float,
        default=1e-3,
        action='store',
        help='Initial learning rate.'
    )
    parser.add_argument(
        '--learning_rate_decay',
        type=float,
        default=0.1,
        action='store',
        help='How much the learning rate decays by after each decay step.'
    )
    parser.add_argument(
        '--learning_rate_decay_step',
        type=float,
        default=25,
        action='store',
        help='How many epochs for a single learning rate decay step.'
    )
    parser.add_argument(
        '--learning_rate_min',
        type=float,
        default=1e-5,
        action='store',
        help='The minimimum learning rate.'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=100,
        action='store',
        help='How many epochs for training.'
    )
    parser.add_argument(
        '--num_readers',
        type=int,
        default=2,
        action='store',
        help='The number of parallel readers to use for reading input.'
    )
    parser.add_argument(
        '--num_preprocessing_threads',
        type=int,
        default=2,
        action='store',
        help='How many threads to use for input preprocessing.'
    )
    parser.add_argument(
        '--num_layers',
        type=int,
        default=1,
        action='store',
        help='How many layers for the rnn cell.'
    )
    parser.add_argument(
        '--num_steps',
        type=int,
        default=2,
        action='store',
        help='How many time steps for the model.'
    )
    parser.add_argument(
        '--num_units',
        type=int,
        default=256,
        action='store',
        help='How many units for the rnn cell.'
    )
    parser.add_argument(
        '--sample_attention_type',
        type=str,
        default='gaussian',
        choices=attention.ATTENTION_TYPES,
        action='store',
        help='The type of sampler to create.'
    )
    parser.add_argument(
        '--sample_type',
        type=str,
        default='normal',
        choices=sample.SAMPLE_TYPES,
        action='store',
        help='The type of sampler to create.'
    )
    parser.add_argument(
        '--sample_size',
        type=int,
        default=20,
        action='store',
        help='The size of the samples from the latent space.'
    )
    parser.add_argument(
        '--samples',
        type=int,
        default=1,
        action='store',
        help='The number of samples from the latent space.'
    )
    parser.add_argument(
        '--summary_frequency',
        type=int,
        default=120,
        action='store',
        help='The number of seconds between running summary operations.'
    )
    parser.add_argument(
        '--trace_frequency',
        type=int,
        default=0,
        action='store',
        help='The number of steps between running trace operations, 0 disables traces.'
    )

    return parser


def _parseable(pair):
    """ Return whether the pair is parseable """
    string_type = isinstance(pair[1], six.string_types)
    is_namespace = isinstance(pair[1], argparse.Namespace)
    iterable_type = isinstance(pair[1], collections.Iterable)
    return (string_type or not iterable_type) and not is_namespace


def _parser(defaults=None):
    """ Return a parser function """
    parser = _json_parser()

    if defaults:
        parser.set_defaults(**defaults)

    def transform_booleans(arg_list):
        """ Handle stringified booleans in the arg list """
        args = []
        for arg in arg_list:
            if arg in ('True', 'False'):
                prefix = '--' if arg == 'True' else '--no-'
                args[-1] = prefix + args[-1][2:]
            else:
                args.append(arg)

        return args

    def parse(pairs):
        """ Validate the passed in json element """
        arg_list = transform_booleans([
            str(item)
            for pair in pairs
            for item in ('--' + pair[0], pair[1])
            if _parseable(pair)])

        parsed = {
            pair[0]: pair[1]
            for pair in pairs
            if not _parseable(pair)}

        if parsed and arg_list:
            raise ValueError('Nested configurations are not allowed')

        return parser.parse_args(arg_list) if arg_list else parsed

    return parse


def override(namespace, overrides):
    """ Override values in namespace with the passed in overrides """
    namespace = vars(namespace)
    namespace.update(overrides)

    return argparse.Namespace(**namespace)


def _overrides(arguments):
    """ Get a dict of overrides """
    parsed = vars(_json_parser().parse_args(arguments))
    arguments = [
        _FLAG_REGEX.match(argument).group(1)
        for argument in arguments
        if _FLAG_REGEX.match(argument)]

    return {argument: parsed[argument] for argument in arguments}


def _load_file(filename, defaults=None):
    """ Load the configuration from a file """
    try:
        stream = open(filename, 'r')
    except Exception:
        raise ValueError('Cannot find config file: {0}'.format(filename))

    return json.load(stream, object_pairs_hook=_parser(defaults=defaults))


def load(arguments, config_file=None, defaults_file=None):
    """ Load a json config """
    defaults = _load_file(defaults_file) if defaults_file else {}
    config = _load_file(config_file, defaults=vars(defaults)) if config_file else {}

    if isinstance(config, dict):
        overrides = _overrides(arguments)
        for (key, namespace) in six.iteritems(config):
            config[key] = override(namespace, overrides)

    return config
