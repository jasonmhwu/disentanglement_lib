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

"""Main training protocol used for semi-supervised disentanglement models."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import gin.tf
import gin.tf.external_configurables  # pylint: disable=unused-import
import numpy as np
import os
import pdb
import pickle
import tensorflow.compat.v1 as tf
import time
from absl import logging
from disentanglement_lib.data.ground_truth import named_data
from disentanglement_lib.methods.semi_supervised import semi_supervised_utils  # pylint: disable=unused-import
from disentanglement_lib.methods.semi_supervised import semi_supervised_vae  # pylint: disable=unused-import
from disentanglement_lib.methods.unsupervised import gaussian_encoder_model
from disentanglement_lib.utils import results
from tensorflow_estimator.python.estimator.tpu import tpu_config
from tensorflow_estimator.python.estimator.tpu.tpu_estimator import TPUEstimator


def train_with_gin(model_dir,
                   overwrite=False,
                   gin_config_files=None,
                   gin_bindings=None):
    """Trains a model based on the provided gin configuration.

    This function will set the provided gin bindings, call the train() function
    and clear the gin config. Please see the train() for required gin bindings.

    Args:
      model_dir: String with path to directory where model output should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      gin_config_files: List of gin config files to load.
      gin_bindings: List of gin bindings to use.
    """
    if gin_config_files is None:
        gin_config_files = []
    if gin_bindings is None:
        gin_bindings = []
    gin_bindings = gin_bindings + [
        "s2_vae.factor_sizes = (3, 6, 40, 32, 32)",
        "s2_factor_vae.factor_sizes = (3, 6, 40, 32, 32)",
        "s2_dip_vae.factor_sizes = (3, 6, 40, 32, 32)",
        "s2_beta_tc_vae.factor_sizes = (3, 6, 40, 32, 32)",
    ]
    gin.parse_config_files_and_bindings(gin_config_files, gin_bindings)
    if gin.query_parameter("dataset.name") != "dsprites_full":
        raise ValueError("S2_beta_VAE isn't configured to train on other datasets yet.")
    train(model_dir, overwrite)
    gin.clear_config()


@gin.configurable("model", denylist=["model_dir"])
def train(model_dir,
          overwrite=False,
          model=gin.REQUIRED,
          training_steps=gin.REQUIRED,
          unsupervised_data_seed=gin.REQUIRED,
          supervised_data_seed=gin.REQUIRED,
          model_seed=gin.REQUIRED,
          batch_size=gin.REQUIRED,
          num_labelled_samples=gin.REQUIRED,
          train_percentage=gin.REQUIRED,
          supervised_sampling_method=gin.REQUIRED,
          supervised_batch_size=gin.REQUIRED,
          name="",
          model_num=None,
          num_checkpoints=10,
          ):
    """Trains the estimator and exports the snapshot and the gin config.

    The use of this function requires the gin binding 'dataset.name' to be
    specified as that determines the data set used for training.

    Args:
      model_dir: String with path to directory where model output should be saved.
      overwrite: Boolean indicating whether to overwrite output directory.
      model: GaussianEncoderModel that should be trained and exported.
      training_steps: Integer with number of training steps.
      unsupervised_data_seed: Integer with random seed used for the unsupervised
        data.
      supervised_data_seed: Integer with random seed for supervised data.
      model_seed: Integer with random seed used for the model.
        batch_size: Integer with the batch size.
      num_labelled_samples: Integer with number of labelled observations for
        training.
      train_percentage: Fraction of the labelled data to use for training (0,1)
      supervised_sampling_method: List of strings about how supervised data are selected.
      supervised_batch_size: batch size of data points to be labelled next iteration.
      name: Optional string with name of the model (can be used to name models).
      num_checkpoints: integer, number of checkpoints spreading evenly acrss training_steps
    """
    # We do not use the variable 'name'. Instead, it can be used to name results
    # as it will be part of the saved gin config.
    del name, model_num

    # Delete the output directory if necessary.
    if tf.gfile.IsDirectory(model_dir):
        if overwrite:
            tf.gfile.DeleteRecursively(model_dir)
        else:
            raise ValueError("Directory already exists and overwrite is False.")
    # Obtain the dataset.
    dataset = named_data.get_named_ground_truth_data()
    labelled_observations = np.concatenate(labelled_observations, 0)
    labelled_factors = np.concatenate(labelled_factors, 0)
    logging.info(f"factor_sizes is {factor_sizes}")
    logging.info(f"labelled_factors shape is {labelled_factors.shape}")
    logging.info(f"labelled_observations shape is {labelled_observations.shape}")

    # We create a TPUEstimator based on the provided model. This is primarily so
    # that we could switch to TPU training in the future. For now, we train
    # locally on GPUs.
    run_config = tpu_config.RunConfig(
        tf_random_seed=model_seed,
        keep_checkpoint_max=num_checkpoints,
        save_checkpoints_steps=training_steps // num_checkpoints,
        tpu_config=tpu_config.TPUConfig(iterations_per_loop=500))
    tpu_estimator = TPUEstimator(
        use_tpu=False,
        model_fn=model.model_fn,
        model_dir=os.path.join(model_dir, "tf_checkpoint"),
        train_batch_size=batch_size,
        eval_batch_size=batch_size,
        config=run_config)

    # Do the actual training.
    assert isinstance(supervised_sampling_method, str)
    experiment_timer = time.time()
    num_iterations = int(np.ceil(num_labelled_samples / supervised_batch_size))
    accumulated_labelled_set_size = 0

    for iter in range(num_iterations):
        # sample points and retrieve ground truth data
        curr_batch_size = min(
            num_labelled_samples - accumulated_labelled_set_size,
            supervised_batch_size
        )
        if iter == 0:  # first batch is always random
            (labelled_observations,
             labelled_factors,
             factor_sizes) = semi_supervised_utils.sample_random_supervised_data(
                supervised_data_seed, dataset, curr_batch_size
            )
        else:
            (new_labelled_observations,
             new_labelled_factors,
             factor_sizes) = semi_supervised_utils.make_supervised_sampler(
                supervised_data_seed, dataset, curr_batch_size, supervised_sampling_method
            )
        accumulated_labelled_set_size += curr_batch_size
        labelled_observations = np.concatenate(
            [labelled_observations, new_labelled_observations],
            axis=0
        )
        labelled_factors = np.concatenate(
            [labelled_factors, new_labelled_factors],
            axis=0
        )
        logging.info(f"{curr_batch_size} points selected from criterion {supervised_sampling_method}")
        logging.info(f"have a total of {accumulated_labelled_set_size} labelled points.")
        logging.info(f"factor_sizes is {factor_sizes}")
        logging.info(f"labelled_factors shape is {labelled_factors.shape}")
        logging.info(f"labelled_observations shape is {labelled_observations.shape}")

        # train
        tpu_estimator.train(
            input_fn=_make_input_fn(
                dataset,
                accumulated_labelled_set_size,
                unsupervised_data_seed,
                labelled_observations,
                labelled_factors,
                train_percentage
            ),
            steps=training_steps // num_iterations
        )
        # TODO: optionally evaluate model performance?

    # Save model as a TFHub module.
    output_shape = named_data.get_named_ground_truth_data().observation_shape
    module_export_path = os.path.join(model_dir, "tfhub")
    gaussian_encoder_model.export_as_tf_hub(model, output_shape,
                                            tpu_estimator.latest_checkpoint(),
                                            module_export_path)

    # Save the results. The result dir will contain all the results and config
    # files that we copied along, as we progress in the pipeline. The idea is that
    # these files will be available for analysis at the end.
    results_dict = tpu_estimator.evaluate(
        input_fn=_make_input_fn(
            dataset,
            num_labelled_samples,
            unsupervised_data_seed,
            labelled_observations,
            labelled_factors,
            train_percentage,
            num_batches=num_labelled_samples,
            validation=True))
    results_dir = os.path.join(model_dir, "results")
    results_dict["elapsed_time"] = time.time() - experiment_timer
    results.update_result_directory(
        results_dir,
        "train_final",
        results_dict,
        create_config=True,
        aggregate_json=True)


def _make_input_fn(ground_truth_data,
                   num_labelled_samples,
                   unsupervised_data_seed,
                   sampled_observations,
                   sampled_factors,
                   train_percentage,
                   num_batches=None,
                   validation=False):
    """Creates an input function for the experiments."""

    def load_dataset(params):
        """TPUEstimator compatible input fuction."""
        dataset = semi_supervised_dataset_from_ground_truth_data(
            ground_truth_data,
            num_labelled_samples,
            unsupervised_data_seed,
            sampled_observations,
            sampled_factors,
            train_percentage,
            validation=validation)
        batch_size = params["batch_size"]
        # We need to drop the remainder as otherwise we lose the batch size in the
        # tensor shape. This has no effect as our data set is infinite.
        dataset = dataset.batch(batch_size, drop_remainder=True)
        if num_batches is not None:
            dataset = dataset.take(num_batches)
        return dataset.make_one_shot_iterator().get_next()

    return load_dataset


def semi_supervised_dataset_from_ground_truth_data(ground_truth_data,
                                                   num_labelled_samples,
                                                   unsupervised_data_seed,
                                                   sampled_observations,
                                                   sampled_factors,
                                                   train_percentage=1.,
                                                   validation=False):
    """Generates a tf.data.DataSet for semi-supervised learning.

    In this setting we have a fixed number of labelled samples and unlimited
    unlabelled samples. The data set will yield of pairs of samples where one
    image is labelled and one is not. Once all the labelled examples are used they
    repeat.

    Args:
      ground_truth_data: Dataset class.
      num_labelled_samples: Number of labelled examples.
      unsupervised_data_seed: Random seed for unsupervised data.
      sampled_observations: Sampled labelled observations.
      sampled_factors: Observed factors of variations for the labelled
        observations.
      train_percentage: Percentage of training points.
      validation: Flag for validation mode.

    Returns:
      tf.data.Dataset, each point is
      (unsupervised observation, (supervised observation, label)).
      For dSprites these are of type
      (np.array(64, 64, 1), (np.array(64, 64, 1), list of length 5.))
    """

    def unsupervised_generator():
        # We need to hard code the random seed for the unsupervised data so that the
        # data set can be reset.
        unsupervised_random_state = np.random.RandomState(unsupervised_data_seed)
        while True:
            yield ground_truth_data.sample_observations(1,
                                                        unsupervised_random_state)[0]

    unlabelled_dataset = tf.data.Dataset.from_generator(
        unsupervised_generator,
        tf.float32,
        output_shapes=ground_truth_data.observation_shape)

    if validation:
        (_, _, sampled_observations,
         sampled_factors) = semi_supervised_utils.train_test_split(
            sampled_observations, sampled_factors, num_labelled_samples,
            train_percentage)
    else:
        (sampled_observations, sampled_factors, _,
         _) = semi_supervised_utils.train_test_split(sampled_observations,
                                                     sampled_factors,
                                                     num_labelled_samples,
                                                     train_percentage)
    labelled_dataset = tf.data.Dataset.from_tensor_slices(
        (tf.to_float(sampled_observations), tf.to_float(sampled_factors)
         )).repeat()
    return tf.data.Dataset.zip((unlabelled_dataset, labelled_dataset))
