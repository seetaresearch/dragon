# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""DLPack utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import workspace
from dragon.core.framework.tensor import Tensor


def from_dlpack(dlpack):
    """Create a tensor sharing the dlpack data.

    Parameters
    ----------
    dlpack : PyCapsule
        The capsule object of a dlpack tensor.

    Returns
    -------
    dragon.Tensor
        The tensor with the dlpack data.

    """
    default_ws = workspace.get_workspace()
    impl = default_ws.create_tensor(scope='DLPack').FromDLPack(dlpack)
    return Tensor(impl=impl, deleter=default_ws._handle_pool)


def to_dlpack(tensor):
    """Return a dlpack tensor sharing the data.

    Parameters
    ----------
    tensor : dragon.Tensor
        The tensor to provide data.

    Returns
    -------
    PyCapsule
        The dlpack tensor.

    """
    return tensor._impl.ToDLPack(tensor.device.to_proto())
