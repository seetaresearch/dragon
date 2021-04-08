# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/saving/hdf5_format.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections

from dragon.core.framework import workspace
from dragon.core.util import logging
from dragon.core.util import six

PICKLE_DEFAULT_PROTOCOL = 2


def load_weights_from_pickle(f, layer, verbose=False):
    default_ws = workspace.get_workspace()
    weight_dict = six.moves.pickle.load(f)
    for weight in layer.weights:
        name = weight.name
        if name in weight_dict:
            value = weight_dict[name]
            value_shape = list(value.shape)
            weight_shape = list(weight.shape)
            if value_shape != weight_shape:
                raise ValueError(
                    'Shape of weight({}) is ({}), \n'
                    'While load from shape of ({}).'
                    .format(name, ', '.join(
                        [str(d) for d in weight_shape]),
                        ', '.join([str(d) for d in value_shape]))
                )
            weight_impl = default_ws.get_tensor(weight.id)
            if weight_impl is not None:
                weight_impl.FromNumpy(value.copy(), True)
                if verbose:
                    logging.info(
                        'Weight({}) loaded, Size: ({})'
                        .format(name, ', '.join([str(d) for d in value_shape])))
            else:
                logging.warning(
                    'Weight({}) is not created '
                    'in current workspace. Skip.'.format(name))


def save_weights_to_pickle(f, layer):
    default_ws = workspace.get_workspace()
    weight_dict = collections.OrderedDict()
    for weight in layer.weights:
        weight_impl = default_ws.get_tensor(weight.id)
        if weight_impl is not None:
            weight_dict[weight.name] = weight_impl.ToNumpy()
    pickle = six.moves.pickle
    pickle.dump(weight_dict, f, PICKLE_DEFAULT_PROTOCOL)
