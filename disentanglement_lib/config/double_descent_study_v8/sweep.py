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

"""Hyperparameter sweeps and configs for the study "unsupervised_study_v1".

Challenging Common Assumptions in the Unsupervised Learning of Disentangled
Representations. Francesco Locatello, Stefan Bauer, Mario Lucic, Gunnar Raetsch,
Sylvain Gelly, Bernhard Schoelkopf, Olivier Bachem. arXiv preprint, 2018.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.config import study
from disentanglement_lib.utils import resources
import disentanglement_lib.utils.hyperparams as h
from six.moves import range


def get_datasets():
    """Returns all the data sets with corresponding correlation indices."""

    # dSprites B
    correlation_indices = h.fixed("correlation_details.corr_indices", [3, 4])
    dataset_name = h.fixed("dataset.name", "dsprites_full")
    config_dsprites = h.zipit([correlation_indices, dataset_name])

    all_datasets = h.chainit([config_dsprites])
    return all_datasets


def get_num_latent():
    return h.sweep("encoder.num_latent", h.categorical([1, 2, 3, 4, 5, 6, 7, 8, 10]))


def get_num_training_data():
    return h.sweep("dataset.num_training_data", h.categorical([1000, 3000, 5000, 7000]))


def get_evaluate_every_n_steps():
    return h.sweep("evaluate_every_n_steps", h.categorical([1000]))


def get_line_biases():
    """Returns random seeds."""
    return h.sweep("correlation_hyperparameter.line_width", h.discrete([10.0]))


def get_seeds(num):
    """Returns random seeds."""
    return h.sweep("model.random_seed", h.categorical(list(range(num))))


def get_num_parameters_scales():
    encoder_scales = h.sweep("conv_encoder.num_parameters_scale", h.discrete([1.0]))
    decoder_scales = h.sweep("deconv_decoder.num_parameters_scale", h.discrete([1.0]))
    return h.zipit([encoder_scales, decoder_scales])


def get_default_models():
    """Our default set of models (6 model * 6 hyperparameters=36 models)."""
    # BetaVAE config.
    model_name = h.fixed("model.name", "beta_vae")
    model_fn = h.fixed("model.model", "@vae()")
    betas = h.sweep("vae.beta", h.discrete([0.125, 1.]))
    config_beta_vae = h.zipit([model_name, betas, model_fn])

    all_models = h.chainit([config_beta_vae])
    return all_models


def get_correlation_types():
    """Returns all types of correlation"""
    return h.sweep(
        "correlation_details.corr_type",
        h.categorical([
            "line"
        ]))


def get_config():
    """Returns the hyperparameter configs for different experiments."""
    arch_enc = h.fixed("encoder.encoder_fn", "@conv_encoder", length=1)
    arch_dec = h.fixed("decoder.decoder_fn", "@deconv_decoder", length=1)
    corr_act = h.fixed("correlation.active_correlation", True)
    architecture = h.zipit([arch_enc, arch_dec, corr_act])
    return h.product([
        get_datasets(),
        get_line_biases(),
        get_correlation_types(),
        architecture,
        get_default_models(),
        get_num_parameters_scales(),
        get_num_latent(),
        get_num_training_data(),
        get_evaluate_every_n_steps(),
        get_seeds(1),
    ])


class DoubleDescentStudyV8(study.Study):
    """Defines the study for the paper."""

    def get_model_config(self, model_num=0):
        """Returns model bindings and config file."""
        config = get_config()[model_num]
        model_bindings = h.to_bindings(config)
        model_config_file = resources.get_file(
            "config/double_descent_study_v8/model_configs/shared.gin")
        return model_bindings, model_config_file

    def get_postprocess_config_files(self):
        """Returns postprocessing config files."""
        return list(
            resources.get_files_in_folder(
                "config/double_descent_study_v8/postprocess_configs/"))

    def get_eval_config_files(self):
        """Returns evaluation config files."""
        return list(
            resources.get_files_in_folder(
                "config/double_descent_study_v8/metric_configs/"))
