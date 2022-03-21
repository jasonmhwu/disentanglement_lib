# coding=utf-8
# Copyright 2018 The DisentanglementLib Authors.  All rights reserved.
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

"""Various utilities used to train and evaluate semi supervised models."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl import logging

import gin.tf
import gin.tf.external_configurables  # pylint: disable=unused-import
import math
import numpy as np
import os
import pickle


# We consider different active learning methods and test their effectiveness
# on boosting disentanglement metrics
@gin.configurable("supervised_sampler")
def make_supervised_sampler(
    supervised_seed,
    ground_truth_data,
    num_labelled_samples,
    sampling_method,
    predictions,
):
    """Wrapper that creates supervised sampler function."""
    if sampling_method == "random":
        return sample_random_supervised_data(
            supervised_seed,
            ground_truth_data,
            num_labelled_samples,
        )
    elif sampling_method == "highest_summed_logvar":
        return highest_summed_logvar(
            predictions["mean"],
            predictions["logvar"],
            ground_truth_data,
            num_labelled_samples,
            num_dims=ground_truth_data.num_factors,
        )
    elif sampling_method == "highest_summed_logvar_all_dims":
        return highest_summed_logvar(
            predictions["mean"],
            predictions["logvar"],
            ground_truth_data,
            num_labelled_samples,
            num_dims=-1
        )
    else:
        raise ValueError("sampling method not implemented")


def sample_random_supervised_data(
    supervised_seed,
    ground_truth_data,
    num_labelled_samples
):
    """Samples data and queries the labeller to obtain labels.

    Args:
      supervised_seed: Seed for the supervised data. Fixing the seed ensures that
        the same data is sampled across different parts of the pipeline.
      ground_truth_data: Dataset class from which the data is to be sampled.
      num_labelled_samples: How many labelled points should be sampled.

    Returns:
      sampled_observations: Numpy array with observations of shape
        (num_labelled_samples, 64, 64, num_channels).
      sampled_factors: Numpy array with observed factors of variations with shape
        (num_labelled_samples, num_factors).
    """
    supervised_random_state = np.random.RandomState(supervised_seed)
    sampled_indices = supervised_random_state.choice(
        ground_truth_data.unlabelled_indices,
        num_labelled_samples,
        replace=False
    )
    ground_truth_data.unlabelled_indices = np.delete(
        ground_truth_data.unlabelled_indices, sampled_indices
    )
    sampled_observations = np.expand_dims(ground_truth_data.images[sampled_indices], 3)
    sampled_factors = ground_truth_data.index_to_factors(sampled_indices)
    sampled_factors, factor_sizes = make_labeller(
        sampled_factors,
        ground_truth_data,
        supervised_random_state
    )
    return sampled_observations, sampled_factors, factor_sizes


def highest_summed_logvar(
    mean, logvar, ground_truth_data, num_labelled_samples, num_dims=-1
):
    """Selects data points with highest logvar summed across latent dimensions.

    Args:
        mean: mean of encoded representations,
            shape of (len(unlabelled_indices), num_latent)
        logvar: lagvar of encoded representations,
            shape of (len(unlabelled_indices), num_latent)
        ground_truth_data: Dataset class from which the data is to be sampled.
        num_labelled_samples: How many labelled points should be sampled.
        num_dims: integer of how many dimensions to consider. if -1, use all dimensions

    Returns:
        sampled_observations: Numpy array with observations of shape
        (num_labelled_samples, 64, 64, num_channels).
      sampled_factors: Numpy array with observed factors of variations with shape
        (num_labelled_samples, num_factors).
    """
    assert mean.shape == (len(ground_truth_data.unlabelled_indices), gin.query_parameter("encoder.num_latent"))
    assert logvar.shape == mean.shape
    del mean

    if num_dims == -1:
        num_dims = logvar.shape[1]
    summed_logvar = np.mean(logvar[:, :num_dims], axis=1)
    logging.info(f"shape of evaluated summed_logvar: {summed_logvar.shape}")
    selected_indices = np.argsort(summed_logvar)[-num_labelled_samples:]
    logging.info(f"selected_indices is {selected_indices}")

    # remove from unlabelled_indices, and create ground truth labels
    ground_truth_data.unlabelled_indices = np.delete(
        ground_truth_data.unlabelled_indices, selected_indices
    )
    selected_observations = np.expand_dims(ground_truth_data.images[selected_indices], 3)
    selected_factors = ground_truth_data.index_to_factors(selected_indices)
    supervised_random_state = np.random.RandomState(0)
    selected_factors, factor_sizes = make_labeller(
        selected_factors,
        ground_truth_data,
        supervised_random_state
    )
    return selected_observations, selected_factors, factor_sizes


def load_supervised_data(
    supervised_seed,
    ground_truth_data,
    num_labelled_samples,
    supervised_selection_criterion
):
    """Load data and queries the labeller to obtain labels.

    Args:
      supervised_seed: Seed for the supervised data. Fixing the seed ensures that
        the same data is sampled across different parts of the pipeline.
      ground_truth_data: Dataset class from which the data is to be sampled.
      num_labelled_samples: How many labelled points should be sampled.
      supervised_selection_criterion: Criterion that the data points are selected from

    Returns:
      sampled_observations: Numpy array with observations of shape
        (num_labelled_samples, 64, 64, num_channels).
      sampled_factors: Numpy array with observed factors of variations with shape
        (num_labelled_samples, num_factors).
    """
    supervised_random_state = np.random.RandomState(supervised_seed)
    pickle_path = os.path.join(
        os.environ.get("DISENTANGLEMENT_LIB_DATA", "."),
        "dsprites", f"{supervised_selection_criterion}.pickle"
    )
    try:
        with open(pickle_path, 'rb') as handle:
            data_points_dict = pickle.load(handle)
        sampled_indices = data_points_dict['informative_indices'][:num_labelled_samples]
        sampled_factors = data_points_dict['informative_factors'][:num_labelled_samples]
        sampled_observations = np.expand_dims(ground_truth_data.images[sampled_indices], 3)
        sampled_factors, factor_sizes = make_labeller(sampled_factors,
                                                      ground_truth_data,
                                                      supervised_random_state)
        return sampled_observations, sampled_factors, factor_sizes
    except FileNotFoundError as e:
        print("can't find pre-made pickle file.")
        print(f"{e}")


def train_test_split(
    observations,
    labels,
    num_labelled_samples,
    train_percentage,
):
    """Splits observations and labels in train and test sets.

    Args:
      observations: Numpy array containing the observations with shape
        (num_labelled_samples, 64, 64, num_channels).
      labels: Numpy array containing the observed factors of variations with shape
        (num_labelled_samples, num_factors).
      num_labelled_samples: How many labelled observations are expected. Used to
        check that the observations have the right shape.
      train_percentage: Float in [0,1] indicating which fraction of the labelled
        data should be used for training.

    Returns:
      observations_train: Numpy array of shape
        (train_percentage * num_labelled_samples, 64, 64, num_channels) containing
        the observations for the training.
      labels_train: Numpy array of shape (train_percentage * num_labelled_samples,
        num_factors) containing the observed factors of variation for the training
        observations.
      observations_test: Numpy array containing the observations for the testing
        with shape ((1-train_percentage) * num_labelled_samples, 64, 64,
          num_channels)
      labels_test: Numpy array containing the observed factors of variation for
        the testing data with shape ((1-train_percentage) * num_labelled_samples,
        num_factors).
    """
    assert observations.shape[0] == num_labelled_samples, \
        "Wrong observations shape."
    num_labelled_samples_train = int(
        math.ceil(num_labelled_samples * train_percentage))
    num_labelled_samples_test = num_labelled_samples - num_labelled_samples_train
    observations_train = observations[:num_labelled_samples_train, :, :, :]
    observations_test = observations[num_labelled_samples_train:, :, :, :]

    labels_train = labels[:num_labelled_samples_train, :]
    labels_test = labels[num_labelled_samples_train:, :]
    assert labels_test.shape[0] == num_labelled_samples_test, "Wrong size test."
    return observations_train, labels_train, observations_test, labels_test


# We consider different oracles (labeller functions) to test the robustness of
# the semi-supervised methods to different types of artifacts in the labels.
@gin.configurable("labeller", denylist=["labels", "dataset"])
def make_labeller(labels,
                  dataset,
                  random_state,
                  labeller_fn=gin.REQUIRED):
    """Wrapper that creates labeller function."""
    return labeller_fn(labels, dataset, random_state)


@gin.configurable(
    "perfect_labeller", denylist=["labels", "dataset"])
def perfect_labeller(labels, dataset, random_state):
    """Returns the true factors of variations without artifacts.

    Args:
      labels: True observations of the factors of variations. Numpy array of shape
        (num_labelled_samples, num_factors) of Float32.
      dataset: Dataset class.
      random_state: Random state for the noise (unused).

    Returns:
      labels: True observations of the factors of variations without artifacts.
        Numpy array of shape (num_labelled_samples, num_factors) of Float32.
    """
    del random_state
    labels = np.float32(labels)
    return labels, dataset.factors_num_values


@gin.configurable("bin_labeller", denylist=["labels", "dataset"])
def bin_labeller(labels, dataset, random_state, num_bins=5):
    """Returns simplified factors of variations.

    The factors of variations are binned to take at most num_bins different values
    to simulate the process of a human roughly labelling the factors of
    variations.

    Args:
      labels: True observations of the factors of variations.
      dataset: Dataset class.
      random_state: Random state for the noise (unused).
      num_bins: Number of bins for the factors of variations.

    Returns:
      labels: Binned factors of variations without noise. Numpy array of shape
        (num_labelled_samples, num_factors) of Float32.
    """
    del random_state
    labels = np.float32(labels)
    for i, num_values in enumerate(dataset.factors_num_values):
        if num_values > num_bins:
            size_bin = (num_values / num_bins)
            labels[:, i] = np.minimum(labels[:, i] // size_bin, num_bins - 1)
    factors_num_values_bin = np.minimum(dataset.factors_num_values, num_bins)
    return labels, factors_num_values_bin


@gin.configurable(
    "noisy_labeller", denylist=["labels", "dataset"])
def noisy_labeller(labels, dataset, random_state, prob_random=0.1):
    """Returns noisy factors of variations.

    With probability prob_random, the observation of the factor of variations is
    uniformly sampled from all possible factor values.

    Args:
      labels: True observations of the factors of variations.
      dataset: Dataset class.
      random_state: Random state for the noise.
      prob_random: Probability of observing random factors of variations.

    Returns:
      labels: Noisy factors of variations. Numpy array of shape
        (num_labelled_samples, num_factors) of Float32.
    """
    for j in range(labels.shape[0]):
        for i, num_values in enumerate(dataset.factors_num_values):
            p = random_state.rand()
            if p < prob_random:
                labels[j, i] = random_state.randint(num_values)
    labels = np.float32(labels)
    return labels, dataset.factors_num_values


@gin.configurable(
    "permuted_labeller", denylist=["labels", "dataset"])
def permuted_labeller(labels, dataset, random_state):
    """Returns factors of variations where the ordinal information is broken.

    Args:
      labels: True observations of the factors of variations.
      dataset: Dataset class.
      random_state: Random state for the noise (unused).

    Returns:
      labels: Noisy factors of variations. Numpy array of shape
        (num_labelled_samples, num_factors) of Float32.
    """
    for i, num_values in enumerate(dataset.factors_num_values):
        labels[:, i] = permute(labels[:, i], num_values, random_state)
    labels = np.float32(labels)
    return labels, dataset.factors_num_values


def permute(factor, num_values, random_state):
    """Permutes the ordinal information of a given factor.

    Args:
      factor: Numpy array with the observations of a factor of varation with shape
        (num_labelled_samples,) and type Int64.
      num_values: Int with number of distinct values the factor of variation can
        take.
      random_state: Random state used to sample the permutation.

    Returns:
      factor: Numpy array of Int64 with the observations of a factor of varation
        with permuted values and shape (num_labelled_samples,).
    """
    unordered_dict = random_state.permutation(range(num_values))
    factor[:] = unordered_dict[factor]
    return factor


def filter_factors(labels, num_observed_factors, random_state):
    """Filter observed factor keeping only a random subset of them.

    Args:
      labels: Factors of variations. Numpy array of shape
        (num_labelled_samples, num_factors) of Float32.
      num_observed_factors: How many factors should be kept.
      random_state: Random state used to sample the permutation.

    Returns:
      Filters the labels so that only num_observed_factors are observed.
    """
    if num_observed_factors < 1:
        raise ValueError("Cannot observe negative amount of factors.")
    elif num_observed_factors > labels.shape[1]:
        raise ValueError(
            "Cannot observe more factors than the ones in the dataset.")
    factors_to_keep = random_state.choice(labels.shape[1],
                                          size=num_observed_factors,
                                          replace=False)
    return labels[:, factors_to_keep], factors_to_keep


@gin.configurable(
    "partial_labeller", denylist=["labels", "dataset"])
def partial_labeller(labels, dataset, random_state,
                     num_observed_factors=2):
    """Returns a few factors of variations without artifacts.

    Args:
      labels: True observations of the factors of variations. Numpy array of shape
        (num_labelled_samples, num_factors) of Float32.
      dataset: Dataset class.
      random_state: Random state for the noise (unused).
      num_observed_factors: How many factors are observed.

    Returns:
      labels: True observations of the factors of variations without artifacts.
        Numpy array of shape (num_labelled_samples, num_factors) of Float32.
    """
    labels = np.float32(labels)
    filtered_factors, factors_to_keep = filter_factors(labels,
                                                       num_observed_factors,
                                                       random_state)

    factors_num_values = [dataset.factors_num_values[i] for i in factors_to_keep]
    return filtered_factors, factors_num_values
