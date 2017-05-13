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

""" Module containing samplers of the latent space """
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.contrib.distributions as distributions
import tensorflow.contrib.framework as framework
import tensorflow.contrib.layers as layers

import glas.model.attention as attentions
import glas.model.rnn as rnn
import glas.utils.graph as graph_utils
from glas.utils.ops import hellinger


SAMPLE_TYPES = ['basic', 'hellinger', 'chisq', 'estimated', 'uniform']


class BasicSampler(rnn.RNN):
    """ The basic latent sampler which uses the reparamaterization trick with a normal distribution
    and calculates latent loss as KL divergence from the standard normal distribution. """
    def __init__(self, config, attention, latent_space, scope='BasicSampler'):
        """ Initialize the sampler """
        super(BasicSampler, self).__init__(scope=scope)

        self.posteriors = []
        self.samples = config.samples
        self.sample_size = config.sample_size

        self.attention = attention
        self.latent_space = latent_space

        shape = (config.batch_size, config.sample_size)
        self.prior = distributions.Normal(tf.zeros(shape), tf.ones(shape), name='prior')

    def compute_moments(self, distribution_or_tensor):
        """ Update the moving averages of the moments based on the passed in tensor """
        if isinstance(distribution_or_tensor, tf.Tensor):
            axes = list(range(distribution_or_tensor.get_shape().ndims - 1))
            return tf.nn.moments(distribution_or_tensor, axes)
        elif isinstance(distribution_or_tensor, distributions.Distribution):
            return distribution_or_tensor.mean(), distribution_or_tensor.variance()
        else:
            raise ValueError('Can only sample a tf.Tensor or distributions.Distribution')

    def approximate_posterior(self, tensor, scope='posterior'):
        """ Calculate the approximate posterior given the tensor """
        # Generate mu and sigma of the Gaussian for the approximate posterior
        with tf.variable_scope(scope, 'posterior', [tensor]):
            mean = layers.linear(tensor, self.sample_size, scope='mean')

            # Use the log of sigma for numerical stability
            log_sigma = layers.linear(tensor, self.sample_size, scope='log_sigma')

            # Create the Gaussian distribution
            sigma = tf.exp(log_sigma)
            posterior = distributions.Normal(mean, sigma, name='posterior')

            self.collect_named_outputs(posterior.loc)
            self.collect_named_outputs(posterior.scale)
            self.posteriors.append(posterior)

            return posterior

    def calculate_latent_loss(self, latent_weights):
        """ Calculate the latent loss in the form of KL divergence """
        for posterior in self.posteriors:
            # NOTE: set allow_nan=True to prevent a CPU-only Assert operation
            kl_divergence = distributions.kl(posterior, self.prior)
            kl_divergence = tf.reduce_sum(latent_weights * kl_divergence, 1, name='kl_divergence')
            tf.losses.add_loss(tf.reduce_mean(kl_divergence, 0, name='kl_divergence/avg'))

    @framework.add_arg_scope
    @rnn.RNN.step_fn
    def random_sample(self, outputs_collections=None):  # pylint: disable=unused-argument
        """ Sample the prior """
        return self.sample(self.prior), None

    def sample(self, distribution_or_tensor, reuse=None):
        """ Sample the passed in distribution or tensor """
        reuse = True if reuse or self.step > 0 else None
        with tf.variable_scope(self.variable_scope, reuse=reuse):
            if isinstance(distribution_or_tensor, tf.Tensor):
                return distribution_or_tensor
            elif isinstance(distribution_or_tensor, distributions.Distribution):
                return tf.reduce_mean(distribution_or_tensor.sample(self.samples), 0)
            else:
                raise ValueError('Can only sample a tf.Tensor or distributions.Distribution')

    def attend(self, tensor):
        """ Use attention over the latent space """
        if self.attention is not None and not isinstance(self.attention, attentions.NoAttention):
            focus = self.attention.read(self.latent_space, tensor)
            tf.add_to_collection(graph_utils.GraphKeys.RNN_OUTPUTS, focus)

            return focus

        return tensor

    @framework.add_arg_scope
    @rnn.RNN.step_fn
    def __call__(self, tensor, outputs_collections=None):
        """ Execute the next time step of the cell """
        focus = self.attend(tensor)
        posterior = self.approximate_posterior(focus)

        sample = self.sample(posterior)
        return sample, None

    next = __call__


class HellingerSampler(BasicSampler):
    """ Latent sampler that uses the Hellinger distance squared rather than the KL divergence. """
    def __init__(self, config, attention, latent_space, scope='HellingerSampler'):
        """ Initialize the sampler """
        super(HellingerSampler, self).__init__(
            config, attention, latent_space, scope=scope)

    def calculate_latent_loss(self, latent_weights):
        """ Calculate the latent loss in the form of KL divergence """
        for posterior in self.posteriors:
            hellinger_distance = latent_weights * hellinger(posterior, self.prior)
            hellinger_distance = tf.reduce_sum(hellinger_distance, 1, name='hellinger')
            tf.losses.add_loss(tf.reduce_mean(hellinger_distance, 0, name='hellinger/avg'))


class MinChiSquaredDistribution(distributions.Distribution):
    """ The minimum chi squared distribution given a mean.

    The prior is assumed to be a standard normal distribution. The probability density function 'f'
    of the minimum chi squared distribution from a prior 'g' given an arithmetic mean is:
        f(x) = g(x) * ((m2_g - m1_f * m1_g) + x * (m1_f - m1_g)) / sigma_g^2

    For more info see section 3.1 from http://web.unbc.ca/~kumarp/d4.pdf """
    def __init__(self,
                 mean,
                 validate_args=False,
                 allow_nan_stats=True,
                 name='MinChiSquaredDistribution'):
        """Construct the mininimum chi squared distribution from the mean. """
        parameters = locals()
        parameters.pop('self')

        with tf.name_scope(name, values=[mean]) as name_scope:
            self._avg = tf.identity(mean, name='mean')
            super(MinChiSquaredDistribution, self).__init__(
                dtype=self._avg.dtype,
                reparameterization_type=distributions.FULLY_REPARAMETERIZED,
                validate_args=validate_args,
                allow_nan_stats=allow_nan_stats,
                parameters=parameters,
                graph_parents=[self._avg],
                name=name_scope)

    @staticmethod
    def _param_shapes(sample_shape):
        return dict(zip(('mean'), ([tf.convert_to_tensor(sample_shape, dtype=tf.int32)])))

    def _batch_shape_tensor(self):
        return tf.shape(self._avg)

    def _batch_shape(self):
        return self._avg.get_shape()

    def _event_shape_tensor(self):
        return tf.constant([], dtype=tf.int32)

    def _event_shape(self):
        return tf.TensorShape([])

    def _sample_n(self, n, seed=None):
        """ Sample the minimum chi squared distribution using reparameterization """
        shape = tf.concat(([n], tf.shape(self.mean())), 0)
        sampled = tf.random_normal(shape=shape, mean=0, stddev=1, dtype=self._avg.dtype, seed=seed)
        return sampled * (1.0 + sampled * self._avg)

    def _log_prob(self, x):
        raise NotImplementedError('Not implemented yet.')

    def _log_cdf(self, x):
        raise NotImplementedError('Not implemented yet.')

    def _cdf(self, x):
        raise NotImplementedError('Not implemented yet.')

    def _log_survival_function(self, x):
        raise NotImplementedError('Not implemented yet.')

    def _survival_function(self, x):
        raise NotImplementedError('Not implemented yet.')

    def _entropy(self):
        raise NotImplementedError('Not implemented yet.')

    def _mean(self):
        return self._avg

    def _variance(self):
        # The variance of the minimum chi squared divergence probability distribution is given
        # by the following (NOTE: mt_f is the t-th moment of the distribution f):
        #   sigma_f^2 = ((m2_g-m1_f*m1_g)*m2_g+(m1_f-m1_g)*m3_g-mu_f^2*sigma_g^2)/sigma_g^2
        #
        # When using the standard normal distribution as the prior 'g' note that:
        #   m1_g = 0, m2_g = 1, m3_g = 0
        #
        # So this becomes:
        #    sigma_f^2 = 1 - mu_f^2
        return 1.0 - tf.square(self._avg)

    def _std(self):
        return tf.sqrt(self._variance())

    def _mode(self):
        return self._avg


class ChiSquaredSampler(BasicSampler):
    """ Latent sampler that uses attention.

    Minimize the chi squared divergence using the first moment of the probability distribution. """
    def __init__(self, config, attention, latent_space, scope='ChiSquaredSampler'):
        """ Initialize the sampler """
        super(ChiSquaredSampler, self).__init__(
            config, attention, latent_space, scope=scope)

        shape = (config.batch_size, self.sample_size)
        self.prior = distributions.Normal(tf.zeros(shape), tf.ones(shape), name='prior')

    def approximate_posterior(self, tensor, scope='posterior'):
        """ Calculate the approximate posterior given the tensor """
        # Generate the minimum chi squared divergence distribution 'f' from the prior 'g'
        with tf.variable_scope(scope, 'posterior', [tensor]):
            mean = layers.linear(tensor, self.sample_size, scope='mean')

            # Create the Gaussian distribution
            posterior = MinChiSquaredDistribution(mean, name='posterior')

            self.collect_named_outputs(posterior.mean())
            self.collect_named_outputs(posterior.variance())
            self.posteriors.append(posterior)

            return posterior

    def calculate_latent_loss(self, latent_weights):
        """ Calculate the latent loss in the form of KL divergence """
        for posterior in self.posteriors:
            # Minimize the chi squared divergence of the posterior 'f' from the prior 'g' (a
            # standard normal distribution), this amounts to minimizing the square of the difference
            # of the first moment of f from the first moment of g divided by the squared variance of
            # g (NOTE: mt_f is the t-th moment of the distribution f):
            #    min(chisq) = (m1_f - m1_g)^2 / sigma_g^2
            #
            # The idea behind using the chi squared divergence is that it is an upper bound for the
            # Kullback-Leibler divergence. The following inequality holds:
            #    KL(f||g) <= log(1 + Chi^2(f||g))
            #
            # So minimize this bound rather than the chi squared divergence directly
            mean, _ = self.compute_moments(posterior)

            axes = tf.range(1, tf.rank(mean))
            chisq = tf.log1p(tf.square(mean - self.prior.mean()) / self.prior.variance())
            chisq = tf.reduce_sum(latent_weights * chisq, axes)
            tf.losses.add_loss(tf.reduce_mean(chisq, name='chisq'))


class EstimatedSampler(ChiSquaredSampler):
    """ Latent sampler that uses attention.

    Estimates the first two moments of the input tensor then uses of the probability integral
    transform assuming the incoming tensor is drawn from a normal distribution F(x), it then
    transforms to a uniform distribution G(x). """
    def __init__(self, config, attention, latent_space, scope='EstimatedSampler'):
        """ Initialize the sampler """
        super(EstimatedSampler, self).__init__(
            config, attention, latent_space, scope=scope)

        shape = (config.batch_size,) + attention.read_size(latent_space)
        self.prior = distributions.Uniform(tf.zeros(shape), tf.ones(shape), name='prior')

    def approximate_posterior(self, tensor, scope='posterior'):
        """ Calculate the approximate posterior given the tensor """
        # Assume the incoming random variable 'X' is drawn from a normal distribution and use the
        # probability integral transform to transform 'X' into 'Y' which is drawn from a standard
        # uniform distribution.
        mean, variance = self.compute_moments(tensor)
        normal = distributions.Normal(mean, tf.sqrt(variance))
        posterior = normal.cdf(tensor)

        self.collect_named_outputs(posterior)
        self.posteriors.append(posterior)

        return posterior


class UniformSampler(EstimatedSampler):
    """ Latent sampler that uses attention.

    Reparameterize the incoming distribution as a uniform distribution specified by with mean and
    variance. """
    def __init__(self, config, attention, latent_space, scope='UniformSampler'):
        """ Initialize the sampler """
        super(UniformSampler, self).__init__(
            config, attention, latent_space, scope=scope)

        shape = (config.batch_size, self.sample_size)
        self.prior = distributions.Uniform(tf.zeros(shape), tf.ones(shape), name='prior')

    def approximate_posterior(self, tensor, scope='posterior'):
        """ Calculate the approximate posterior given the tensor """
        # Generate mu and sigma of the Gaussian for the approximate posterior
        sample_size = self.prior.batch_shape.as_list()[-1]
        with tf.variable_scope(scope, 'posterior', [tensor]):
            mean = layers.linear(tensor, sample_size, scope='mean')

            # Use the log of sigma for numerical stability
            log_variance = layers.linear(tensor, sample_size, scope='log_variance')

            # Create the Uniform distribution
            variance = tf.exp(log_variance)
            delta = tf.sqrt(3.0 * variance)
            posterior = distributions.Uniform(mean - delta, mean + delta, name='posterior')

            self.collect_named_outputs(posterior.low)
            self.collect_named_outputs(posterior.high)
            self.posteriors.append(posterior)

            return posterior


def create_latent_space(batch_size, shape, steps=None):
    """ Create the latent space """
    # Setup the latent space. The latent space is a 2-D tensor used for each element in the batch
    # with dimensions [batch_size, latent_size, latent_size]. If steps are provided then there is a
    # latent space per step with dimensions [step, batch_size, latent_size, latent_size].
    latent_shape = shape
    if steps is not None:
        latent_shape = (steps,) + latent_shape

    latent_space = framework.model_variable(
        'LatentSpace', shape=latent_shape, trainable=True,
        initializer=tf.random_uniform_initializer(0.0, 1e-3))
    latent_space = tf.tile(latent_space, (batch_size,) + (1,) * (len(latent_shape) - 1))
    latent_space = tf.reshape(latent_space, (batch_size,) + latent_shape)

    if steps is not None:
        permutation = (1, 0) + tuple(x + 2 for x in range(len(shape)))
        latent_space = tf.transpose(latent_space, permutation)

    return latent_space


def create_sampler(config):
    """ Create the appropriate sampler based on the passed in config """
    if config.sample_type == 'basic':
        sample_type = BasicSampler
    elif config.sample_type == 'hellinger':
        sample_type = HellingerSampler
    elif config.sample_type == 'chisq':
        sample_type = ChiSquaredSampler
    elif config.sample_type == 'estimated':
        sample_type = EstimatedSampler
    elif config.sample_type == 'uniform':
        sample_type = UniformSampler

    latent_size = (config.latent_size, config.latent_size)
    latent_space = create_latent_space(config.batch_size, latent_size)
    attention = attentions.create_attention(
        config.sample_attention_type, latent_size, read_size=config.latent_read_size)

    return sample_type(config, attention, latent_space)
