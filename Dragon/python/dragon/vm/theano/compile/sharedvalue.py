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

import numpy as np
import dragon as dg


def shared(value, name=None, **kwargs):
    """Construct a Tensor initialized with ``value``.

    Parameters
    ----------
    value : number, list or numpy.ndarray
        The numerical values.
    name : str
        The name of tensor.

    Returns
    -------
    Tensor
        The initialized tensor.

    """
    if not isinstance(value, (int, float, list, np.ndarray)):
        raise TypeError("Unsupported type of value: {}".format(type(value)))
    tensor = dg.Tensor(name).Variable()
    dg.workspace.FeedTensor(tensor, value)
    return tensor