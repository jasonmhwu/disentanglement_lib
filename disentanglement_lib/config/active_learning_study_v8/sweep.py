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
  """Returns all the data sets."""
  return h.sweep(
      "dataset.name",
      h.categorical([
          "dsprites_full"
      ]))


def get_num_latent(sweep):
  return h.sweep("encoder.num_latent", h.discrete(sweep))


def get_seeds(num):
  """Returns random seeds."""
  return h.sweep("model.model_seed", h.categorical(list(range(num))))


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
  parameters = h.product([betas, gamma_sups])
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
          "highest_summed_logvar",
          "highest_summed_logvar_all_dims",
      ]))


def get_config():      
  """Returns the hyperparameter configs for different experiments."""
  arch_enc = h.fixed("encoder.encoder_fn", "@conv_encoder", length=1)
  arch_dec = h.fixed("decoder.decoder_fn", "@deconv_decoder", length=1)
  architecture = h.zipit([arch_enc, arch_dec])
  return h.product([
      get_datasets(),
      architecture,
      get_default_models(),
      get_supervised_sampling_method(),
      get_seeds(1),
  ])


class ActiveLearningStudyV8(study.Study):
  """Defines the study for the paper."""
  def __init__(self):
    self.study_name = "active_learning_study_v8"

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
