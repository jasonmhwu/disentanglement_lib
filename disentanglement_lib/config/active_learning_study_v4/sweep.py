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
  betas = h.sweep("s2_vae.beta", h.discrete([1., 4., 16., 32., 48.]))
  gamma_sups = h.sweep("s2_vae.gamma_sup", h.discrete([1., 4.]))
  parameters = h.product([betas, gamma_sups])
  config_s2_beta_vae = h.zipit([model_name, parameters, model_fn])
 
  # Semi-Supervised FactorVAE config
  model_name = h.fixed("model.name", "s2_factor_vae")
  model_fn = h.fixed("model.model", "@s2_factor_vae()")
  discr_fn = h.fixed("discriminator.discriminator_fn", "@fc_discriminator")
  betas = h.sweep("s2_factor_vae.gamma", h.discrete([10., 30., 50., 100., 200.]))
  gamma_sups = h.sweep("s2_factor_vae.gamma_sup", h.discrete([10., 50.]))
  parameters = h.product([betas, gamma_sups])
  config_s2_factor_vae = h.zipit([model_name, parameters, model_fn, discr_fn])
  
  # Semi-Supervised DIP-VAE-1 config
  model_name = h.fixed("model.name", "s2_dip_vae_1")
  model_fn = h.fixed("model.model", "@s2_dip_vae()")
  lambda_od = h.sweep("s2_dip_vae.lambda_od",
                      h.discrete([1., 5., 20., 50., 100.]))
  lambda_d_factor = h.fixed("s2_dip_vae.lambda_d_factor", 10.)
  dip_type = h.fixed("dip_vae.dip_type", "i")
  gamma_sups = h.sweep("s2_dip_vae.gamma_sup", h.discrete([10., 50.]))
  parameters = h.product([lambda_od, lambda_d_factor, dip_type, gamma_sups])
  config_s2_dip_vae = h.zipit([model_name, parameters, model_fn])
  
  # Semi-Supervised TCVAE config.
  model_name = h.fixed("model.name", "s2_beta_tc_vae")
  model_fn = h.fixed("model.model", "@s2_beta_tc_vae()")
  betas = h.sweep("s2_beta_tc_vae.beta", h.discrete([1., 4., 16., 32., 48.]))
  gamma_sups = h.sweep("s2_beta_tc_vae.gamma_sup", h.discrete([1., 4.]))
  parameters = h.product([betas, gamma_sups])
  config_s2_beta_tc_vae = h.zipit([model_name, parameters, model_fn])
 
  all_models = h.chainit([
      config_s2_beta_vae, config_s2_factor_vae, config_s2_dip_vae, config_s2_beta_tc_vae,
  ])
  return all_models


def get_supervised_selection_criterion():
  """Returns all supervised selection criterion."""
  return h.sweep(
      "model.supervised_selection_criterion",
      h.categorical([
          "highest_summed_std",
          "lowest_summed_std",
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
      get_supervised_selection_criterion(),
      get_seeds(1),
  ])


class ActiveLearningStudyV4(study.Study):
  """Defines the study for the paper."""
  def __init__(self):
    self.study_name = "active_learning_study_v4"

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
