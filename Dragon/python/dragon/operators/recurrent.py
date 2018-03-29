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


def LSTMUnit(c_t_1, gate_input, cont_t=None, **kwargs):
    """Simple LSTMCell module.

    Parameters
    ----------
    c_t_1 : Tensor
        The initial state of cell.
    gate_input : Tensor
        The concatenated input for 4 gates.
    cont_t : Tensor
        The mask to discard specific instances. Default is ``None``.

    Returns
    -------
    tuple
        The lstm outputs, represent ``c`` and ``h`` respectively.

    """
    arguments = ParseArguments(locals())
    if cont_t is not None:
        if isinstance(cont_t, Tensor):
            raise TypeError('The tyoe of cont_t should Tensor.')
        arguments['cont_t'] = cont_t.name
    return Tensor.CreateOperator(inputs=[c_t_1, gate_input], nout=2,
                                 op_type='LSTMUnit', **arguments)