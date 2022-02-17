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

"""Provides named, gin configurable ground truth data sets."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from disentanglement_lib.data.ground_truth import cars3d
from disentanglement_lib.data.ground_truth import dsprites
from disentanglement_lib.data.ground_truth import dummy_data
from disentanglement_lib.data.ground_truth import mpi3d
from disentanglement_lib.data.ground_truth import norb
from disentanglement_lib.data.ground_truth import shapes3d
import gin.tf
from absl import logging


@gin.configurable("dataset")
def get_named_ground_truth_data(
  name,
  num_training_data,
  train_with_full_dataset='True',
  split_method='all',
):
  """Returns ground truth data set based on name.

  Args:
    name: String with the name of the dataset.
    train_with_full_dataset: bool, whether to use full dataset for training.
    split_method: "train" or "valid", only used when train_with_full_dataset is False.
    num_training_data: number of data points used in training dataset

  Raises:
    ValueError: if an invalid data set name is provided.
  """
  logging.info(f"dataset.name: {name}")
  logging.info(f"dataset.num_training_data: {num_training_data}")
  logging.info(f"dataset.split_method: {split_method}")
  logging.info(f"dataset.train_with_full_dataset: {train_with_full_dataset}")
  if (train_with_full_dataset == 'False') and (name == "dsprites_full"):
    assert split_method in ['train', 'valid']
    logging.info(f"dsprites subset has {num_training_data} data points.")
    return dsprites.SubsetDSprites(
      [1, 2, 3, 4, 5],
      split_method=split_method,
      num_training_data=num_training_data,
    )

  assert train_with_full_dataset == 'True'

  if name == "dsprites_full":
    return dsprites.DSprites([1, 2, 3, 4, 5])
  elif name == "dsprites_noshape":
    return dsprites.DSprites([2, 3, 4, 5])
  elif name == "color_dsprites":
    return dsprites.ColorDSprites([1, 2, 3, 4, 5])
  elif name == "noisy_dsprites":
    return dsprites.NoisyDSprites([1, 2, 3, 4, 5])
  elif name == "scream_dsprites":
    return dsprites.ScreamDSprites([1, 2, 3, 4, 5])
  elif name == "smallnorb":
    return norb.SmallNORB()
  elif name == "cars3d":
    return cars3d.Cars3D()
  elif name == "mpi3d_toy":
    return mpi3d.MPI3D(mode="mpi3d_toy")
  elif name == "mpi3d_realistic":
    return mpi3d.MPI3D(mode="mpi3d_realistic")
  elif name == "mpi3d_real":
    return mpi3d.MPI3D(mode="mpi3d_real")
  elif name == "shapes3d":
    return shapes3d.Shapes3D()
  elif name == "dummy_data":
    return dummy_data.DummyData()
  else:
    raise ValueError("Invalid data set name.")
