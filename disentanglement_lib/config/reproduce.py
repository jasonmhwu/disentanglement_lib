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

"""Different studies that can be reproduced."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from disentanglement_lib.config.abstract_reasoning_study_v1.stage1 import sweep as abstract_reasoning_study_v1
from disentanglement_lib.config.fairness_study_v1 import sweep as fairness_study_v1
from disentanglement_lib.config.tests import sweep as tests
from disentanglement_lib.config.unsupervised_study_v1 import sweep as unsupervised_study_v1
from disentanglement_lib.config.correlated_factors_study import sweep as correlated_factors_study
from disentanglement_lib.config.correlated_factors_study_ws_od import sweep as correlated_factors_study_ws_od
from disentanglement_lib.config.correlated_factors_study_ws_id1 import sweep as correlated_factors_study_ws_id1
from disentanglement_lib.config.correlated_factors_study_ws_id2 import sweep as correlated_factors_study_ws_id2
from disentanglement_lib.config.double_descent_study import sweep as double_descent_study
# from disentanglement_lib.config.double_descent_study_v2 import sweep as double_descent_study_v2
# from disentanglement_lib.config.double_descent_study_v3 import sweep as double_descent_study_v3
# from disentanglement_lib.config.double_descent_study_v4 import sweep as double_descent_study_v4
# from disentanglement_lib.config.double_descent_study_v5 import sweep as double_descent_study_v5
# from disentanglement_lib.config.double_descent_study_v6 import sweep as double_descent_study_v6
# from disentanglement_lib.config.double_descent_study_v7 import sweep as double_descent_study_v7
# from disentanglement_lib.config.double_descent_study_v8 import sweep as double_descent_study_v8
# from disentanglement_lib.config.double_descent_study_v9 import sweep as double_descent_study_v9
# from disentanglement_lib.config.double_descent_study_v10 import sweep as double_descent_study_v10

# from disentanglement_lib.config.active_learning_study_v1 import sweep as active_learning_study_v1
# from disentanglement_lib.config.active_learning_study_v2 import sweep as active_learning_study_v2
# from disentanglement_lib.config.active_learning_study_v3 import sweep as active_learning_study_v3
# from disentanglement_lib.config.active_learning_study_v4 import sweep as active_learning_study_v4
# from disentanglement_lib.config.active_learning_study_v5 import sweep as active_learning_study_v5
# from disentanglement_lib.config.active_learning_study_v6 import sweep as active_learning_study_v6
from disentanglement_lib.config.active_learning_study_v7 import sweep as active_learning_study_v7
from disentanglement_lib.config.active_learning_study_v8 import sweep as active_learning_study_v8
from disentanglement_lib.config.active_learning_study_v9 import sweep as active_learning_study_v9
from disentanglement_lib.config.active_learning_study_v10 import sweep as active_learning_study_v10
from disentanglement_lib.config.active_learning_study_v11 import sweep as active_learning_study_v11
from disentanglement_lib.config.active_learning_study_v12 import sweep as active_learning_study_v12
from disentanglement_lib.config.active_learning_study_v13 import sweep as active_learning_study_v13
from disentanglement_lib.config.active_learning_study_v14 import sweep as active_learning_study_v14
from disentanglement_lib.config.active_learning_study_v15 import sweep as active_learning_study_v15


STUDIES = {
    "unsupervised_study_v1": unsupervised_study_v1.UnsupervisedStudyV1(),
    "abstract_reasoning_study_v1":
        abstract_reasoning_study_v1.AbstractReasoningStudyV1(),
    "fairness_study_v1":
        fairness_study_v1.FairnessStudyV1(),
    "correlation_study":
        correlated_factors_study.CorrelatedFactorsStudy(),
    "correlation_study_ws_od":
        correlated_factors_study_ws_od.CorrelatedFactorsStudyWSOD(),
    "correlation_study_ws_id1":
        correlated_factors_study_ws_id1.CorrelatedFactorsStudyWSID1(),
    "correlation_study_ws_id2":
        correlated_factors_study_ws_id2.CorrelatedFactorsStudyWSID2(),
    "double_descent_study":
        double_descent_study.DoubleDescentStudy(),
    "test": tests.TestStudy(),
    "active_learning_study_v7": active_learning_study_v7.ActiveLearningStudyV7(),
    "active_learning_study_v8": active_learning_study_v8.ActiveLearningStudyV8(),
    "active_learning_study_v9": active_learning_study_v9.ActiveLearningStudyV9(),
    "active_learning_study_v10": active_learning_study_v10.ActiveLearningStudyV10(),
    "active_learning_study_v11": active_learning_study_v11.ActiveLearningStudyV11(),
    "active_learning_study_v12": active_learning_study_v12.ActiveLearningStudyV12(),
    "active_learning_study_v13": active_learning_study_v13.ActiveLearningStudyV13(),
    "active_learning_study_v14": active_learning_study_v14.ActiveLearningStudyV14(),
    "active_learning_study_v15": active_learning_study_v15.ActiveLearningStudyV15(),
}
