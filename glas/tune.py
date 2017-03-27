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

""" Main entry point for tuning hyperparameters of the GLAS model. """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import functools
from itertools import chain as iterchain
import json
import logging
import pickle
import os
import random
import re
import shutil
import subprocess

from six.moves import xrange  # pylint: disable=redefined-builtin
from six import iteritems

from glas.eval import __file__ as eval_script
from glas.train import __file__ as train_script


_LOSS_REGEX = re.compile(r'.*total_loss\[([\d.]+)\]', re.DOTALL)
_HYPERPARAMETER_SELECTORS = {
    'latent_read_size': functools.partial(random.randint, 1, 16),
    'latent_size': functools.partial(random.randint, 8, 128),
    'sample_size': functools.partial(random.randint, 1, 256),
}


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
        '--folds',
        type=int,
        default=2,
        action='store',
        help='How many folds to use for cross validation'
    )
    parser.add_argument(
        '--log_level',
        type=str,
        default='INFO',
        choices=['DEBUG', 'ERROR', 'CRITICAL', 'INFO', 'WARN'],
        action='store',
        help='JSON config file for the argument defaults'
    )
    parser.add_argument(
        '--log_dir',
        type=str,
        default='tune',
        action='store',
        help='The log directory.'
    )
    parser.add_argument(
        '--max_evals',
        type=int,
        default=10,
        action='store',
        help='Maximum number of times to evaluate the model'
    )
    parser.add_argument(
        '--num_epochs',
        type=int,
        default=1,
        action='store',
        help='Override for the num epochs for each trial'
    )

    return parser.parse_args()


def select_hyperparameters():
    """ Select a new hyperparameter value based on the previous values """
    hyperparameters = {param: func() for (param, func) in iteritems(_HYPERPARAMETER_SELECTORS)}

    # Ensure latent read size is not bigger than the latent size
    hyperparameters['latent_read_size'] = min(
        hyperparameters['latent_size'],
        hyperparameters['latent_read_size'])

    logging.info('Selected hyperparameters: %s', hyperparameters)

    return hyperparameters


def generate_command_args(overrides, num_folds, fold):
    """ Create the command line with the passed in config overrides """
    args = []
    args.append('--config_file')
    args.append(FLAGS.config_file)
    args.append('--defaults_file')
    args.append(FLAGS.defaults_file)
    args.extend(iterchain.from_iterable([('--' + str(k), str(v)) for k, v in iteritems(overrides)]))

    # Setup the folds for cross validation
    args.append('--datasubset')
    args.append('train')
    args.append('--fold_count')
    args.append(str(num_folds))
    args.append('--fold')
    args.append(str(fold))

    return args


def training_directory(trial_index, fold):
    """ Get the training directory given the trial index """
    return os.path.join(FLAGS.log_dir, 'train{0}-fold{1}'.format(trial_index, fold))


def eval_directory(trial_index, fold):
    """ Get the evaluation directory given the trial index """
    return os.path.join(FLAGS.log_dir, 'eval{0}-fold{1}'.format(trial_index, fold))


def tuning_filename():
    """ Return the filename used for storing hyperparameter tuning information """
    return os.path.join(FLAGS.log_dir, 'tuning.json')


def cleanup_trials(trials):
    """ Clean up any in progress trials """
    # If a directory exists whose index is the same as the length of the trails list, then it must
    # not have completed.
    index = len(trials)
    for fold in xrange(FLAGS.folds):
        directory = eval_directory(index, fold)
        if os.path.isdir(directory):
            shutil.rmtree(directory)

        directory = training_directory(index, fold)
        if os.path.isdir(directory):
            shutil.rmtree(directory)


def load_tuning():
    """ Load any previous tuning information """
    filename = tuning_filename()
    try:
        stream = open(filename, 'r')
    except IOError:
        logging.info('No previous tuning file found.')
        return {'trials': []}

    tuning = json.load(stream)
    random.setstate(pickle.loads(tuning['rng_state']))

    cleanup_trials(tuning['trials'])
    return tuning


def save_tuning(tuning):
    """ Save the current tuning information """
    filename = tuning_filename()
    try:
        stream = open(filename, 'w')
    except IOError:
        logging.error('Cannot save tuning file.')

    tuning['rng_state'] = pickle.dumps(random.getstate())
    return json.dump(tuning, stream)


def train(trial_index, fold, hyperparameters):
    """ Train using the passed in hyperparameters """
    args = ['python', train_script]
    args.extend(generate_command_args(hyperparameters, FLAGS.folds, fold))

    args.append('--num_epochs')
    args.append(str(FLAGS.num_epochs))
    args.append('--log_dir')
    args.append(training_directory(trial_index, fold))

    logging.debug('Executing: %s', args)

    try:
        subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as cae:
        raise cae


def evaluate(trial_index, fold, hyperparameters):
    """ Evaluate using the passed in hyperparameters """
    args = ['python', eval_script]
    args.extend(generate_command_args(hyperparameters, FLAGS.folds, fold))

    args.append('--once')
    args.append('--checkpoint_path')
    args.append(training_directory(trial_index, fold))
    args.append('--log_dir')
    args.append(eval_directory(trial_index, fold))

    logging.debug('Executing: %s', args)

    try:
        output = subprocess.check_output(args, stderr=subprocess.STDOUT)
    except subprocess.CalledProcessError as cae:
        raise cae

    return float(_LOSS_REGEX.match(output).group(1))


def tune_model(tuning):
    """ Tune hyperparameters of the model using random search """
    trials = tuning['trials']
    for trial_index in xrange(len(trials), FLAGS.max_evals):
        logging.info('Beginning trial #%d', trial_index)
        hyperparameters = select_hyperparameters()

        loss = 0.0
        for fold in xrange(FLAGS.folds):
            train(trial_index, fold, hyperparameters)
            loss += evaluate(trial_index, fold, hyperparameters)

        trial = {'loss': loss / FLAGS.folds, 'hyperparameters': hyperparameters}
        logging.info('Completed trial #%d: %s', trial_index, trial)

        trials.append(trial)
        save_tuning(tuning)

    return tuning


def main():
    """ The main entry point """
    tuning = load_tuning()
    tuning = tune_model(tuning)

    best_trial = {}
    trials = tuning['trials']
    for trial in trials:
        if not best_trial or float(trial['loss']) < float(best_trial['loss']):
            best_trial = trial

    print('Best trial: {0}'.format(best_trial))


if __name__ == '__main__':
    FLAGS = parse_commandline()
    logging.basicConfig(level=FLAGS.log_level)
    main()
