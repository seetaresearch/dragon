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

from dragon.core.tensor import Tensor as _Tensor


def shared(value, name=None, **kwargs):
    """Construct a Tensor initialized with ``value``.

    Parameters
    ----------
    value : number, sequence or numpy.ndarray
        The numerical values.
    name : str, optional
        The optional name

    Returns
    -------
    Tensor
        The initialized tensor.

    """
    return _Tensor(name).set_value(value)