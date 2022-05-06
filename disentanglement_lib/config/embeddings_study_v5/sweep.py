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

import disentanglement_lib.utils.hyperparams as h
from disentanglement_lib.config import study
from disentanglement_lib.utils import resources
from six.moves import range


def get_datasets():
    """Returns all the data sets."""
    # make my choice of datasets here
    datasets = ["mpi3d_toy", "cars3d", "shapes3d"]

    dataset_factor_sizes = {
        "dsprites_full": (3, 6, 40, 32, 32),
        "manual_dsprites": (3, 6, 40, 32, 32),
        "smallnorb": (5, 9, 18, 6),
        "mpi3d_toy": (4, 4, 2, 3, 3, 40, 40),
        "shapes3d": (10, 10, 10, 8, 4, 15),
        "cars3d": (4, 24, 183),
    }
    dataset_embedding_dimensions = {
        "manual_dsprites": (1, 1, 2, 1, 1),
        "mpi3d_toy": (2, 2, 1, 1, 1, 1, 1),
        "cars3d": (1, 2, 3),
        "shapes3d": (2, 2, 2, 1, 2, 1),
    }
    dataset_supervised_loss = {
        "mpi3d_toy": ("multidimensional_embed", "multidimensional_embed", "xent", "xent", "xent", "xent", "xent"),
        "cars3d": ("xent", "multidimensional_embed", "multidimensional_embed"),
        "shapes3d": ("multidimensional_embed", "multidimensional_embed", "multidimensional_embed",
            "xent", "multidimensional_embed", "xent"),
    }

    configs = []
    for dataset in datasets:
        dataset_name = h.fixed("dataset.name", dataset)
        s2_vae_param = h.fixed("s2_vae.factor_sizes", dataset_factor_sizes[dataset])
        s2_factor_vae_param = h.fixed("s2_factor_vae.factor_sizes", dataset_factor_sizes[dataset])
        s2_dip_vae_param = h.fixed("s2_dip_vae.factor_sizes", dataset_factor_sizes[dataset])
        s2_beta_tc_vae_param = h.fixed("s2_beta_tc_vae.factor_sizes", dataset_factor_sizes[dataset])
        s2_independent_vae_param = h.fixed("s2_independent_vae.factor_sizes", dataset_factor_sizes[dataset])
        # add observed factors
        num_factors = len(dataset_factor_sizes[dataset])
        observed_factors_list = [list(range(num_factors))]
        embedding_dimensions = []
        loss_terms = []
        for factors_indices in observed_factors_list:
            embedding_dimensions.append([dataset_embedding_dimensions[dataset][i] for i in factors_indices])
            loss_terms.append([dataset_supervised_loss[dataset][i] for i in factors_indices])
        observed_factors = h.sweep(
            "fixed_partial_labeller.observed_factor_indices",
            h.discrete(observed_factors_list))
        embedding_dims = h.sweep(
            "multidimensional_embed.embedding_dimensions",
            h.discrete(embedding_dimensions))
        supervised_info = h.zipit([observed_factors, embedding_dims])

        configs.append(h.product([
            dataset_name,
            s2_vae_param,
            s2_factor_vae_param,
            s2_dip_vae_param,
            s2_beta_tc_vae_param,
            s2_independent_vae_param,
            supervised_info,
        ]))
    return h.chainit(configs)


def get_num_latent(sweep):
    return h.sweep("encoder.num_latent", h.discrete(sweep))


def get_seeds(num, start=0):
    """Returns random seeds."""
    return h.sweep("model.model_seed", h.categorical(list(range(start, num))))


def get_default_models():
    """Our default set of models (6 model * 6 hyperparameters=36 models)."""
    # BetaVAE config.
    # model_name = h.fixed("model.name", "beta_vae")
    # model_fn = h.fixed("model.model", "@vae()")
    # betas = h.sweep("vae.beta", h.discrete([1., 2., 4., 6., 8., 16.]))
    # config_beta_vae = h.zipit([model_name, betas, model_fn])

    # Semi-Supervised BetaVAE config.
    model_name = h.fixed("model.name", "s2_beta_vae")
    model_fn = h.fixed("model.model", "@s2_vae()")
    betas = h.sweep("s2_vae.beta", h.discrete([1., 4., 16.]))
    gamma_sups = h.sweep("s2_vae.gamma_sup", h.discrete([1.]))
    num_stochastic_passes = h.sweep("s2_vae.num_stochastic_passes", h.discrete([1]))
    parameters = h.product([betas, gamma_sups, num_stochastic_passes])
    config_s2_beta_vae = h.zipit([model_name, parameters, model_fn])

    all_models = h.chainit([
        config_s2_beta_vae
    ])
    return all_models


def get_supervised_sampling_method():
    """Returns all supervised sampling methods."""
    return h.sweep(
        "model.supervised_sampling_method",
        h.categorical([
            "random",
        ]))


def get_uncertainty_method():
    """Returns all uncertainty methods."""
    return h.sweep(
        "model.uncertainty_method",
        h.categorical([
            "dropout_mean",
        ]))


def get_supervised_loss():
    """Returns loss functions for supervised loss."""
    loss_fn = h.sweep(
        "supervised_loss.loss_fn",
        h.discrete(['multidimensional_embed']),
    )
    sigma = h.fixed("multidimensional_embed.sigma", "learn")
    return h.product([loss_fn, sigma])

def get_config():
    """Returns the hyperparameter configs for different experiments."""
    arch_enc = h.fixed("encoder.encoder_fn", "@conv_encoder", length=1)
    arch_dec = h.fixed("decoder.decoder_fn", "@deconv_decoder", length=1)
    architecture = h.zipit([arch_enc, arch_dec])
    return h.product([
        get_seeds(5, start=0),
        get_datasets(),
        architecture,
        get_default_models(),
        get_supervised_sampling_method(),
        get_uncertainty_method(),
        get_supervised_loss(),
    ])


class EmbeddingsStudyV5(study.Study):
    """Defines the study for the paper."""

    def __init__(self):
        self.study_name = "embeddings_study_v5"

    def get_model_config(self, model_num=0):
        """Returns model bindings and config file."""
        config = get_config()[model_num]
        model_bindings = h.to_bindings(config)
        model_config_file = resources.get_file(
            f"config/{self.study_name}/model_configs/shared.gin")
        return model_bindings, model_config_file

    def get_postprocess_config_files(self):
        """Returns postprocessing config files."""
        return list(
            resources.get_files_in_folder(
                f"config/{self.study_name}/postprocess_configs/"))

    def get_eval_config_files(self):
        """Returns evaluation config files."""
        return list(
            resources.get_files_in_folder(
                f"config/{self.study_name}/metric_configs/"))
