# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .. import *


def RNNParamSet(
    inputs, layer_id, param_id, param_type,
        rnn_mode, input_size, hidden_size,
            num_layers=1, num_directions=1, **kwargs
):
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())
    arguments['inputs'] = inputs[1]
    arguments['existing_outputs'] = inputs[0]
    return Tensor.CreateOperator(op_type='RNNParamSet', **arguments)