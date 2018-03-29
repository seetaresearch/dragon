# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
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

from . import *


def FloatToHalf(inputs, **kwargs):
    """Cast the type of tensor from ``float32`` to ``float16``.

    Parameters
    ----------
    inputs : Tensor
        The ``float32`` tensor.

    Returns
    -------
    Tensor
        The ``float16`` tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    output = Tensor.CreateOperator(nout=1, op_type='FloatToHalf', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output