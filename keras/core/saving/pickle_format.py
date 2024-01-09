# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Pickle utilities."""

import collections
import pickle

from dragon.core.framework import workspace
from dragon.core.util import logging

PICKLE_DEFAULT_PROTOCOL = 2


def load_weights_from_pickle(f, layer, verbose=False):
    default_ws = workspace.get_workspace()
    weight_dict = pickle.load(f)
    for weight in layer.weights:
        name = weight.name
        if name in weight_dict:
            value = weight_dict[name]
            value_shape = list(value.shape)
            weight_shape = list(weight.shape)
            if value_shape != weight_shape:
                raise ValueError(
                    "Shape of weight({}) is ({}), \n"
                    "While load from shape of ({}).".format(
                        name,
                        ", ".join([str(d) for d in weight_shape]),
                        ", ".join([str(d) for d in value_shape]),
                    )
                )
            weight_impl = default_ws.get_tensor(weight.id)
            if weight_impl is not None:
                weight_impl.FromNumpy(value.copy(), True)
                if verbose:
                    logging.info(
                        "Weight({}) loaded, Size: ({})".format(
                            name, ", ".join([str(d) for d in value_shape])
                        )
                    )
            else:
                logging.warning(
                    "Weight({}) is not created " "in current workspace. Skip.".format(name)
                )


def save_weights_to_pickle(f, layer):
    default_ws = workspace.get_workspace()
    weight_dict = collections.OrderedDict()
    for weight in layer.weights:
        weight_impl = default_ws.get_tensor(weight.id)
        if weight_impl is not None:
            weight_dict[weight.name] = weight_impl.ToNumpy()
    pickle.dump(weight_dict, f, PICKLE_DEFAULT_PROTOCOL)
