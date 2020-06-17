# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.eager.tensor import EagerTensor
from dragon.core.framework import device_spec
from dragon.core.framework import workspace


def from_dlpack(dlpack):
    """Create a tensor sharing the dlpack data.

    Parameters
    ----------
    dlpack : PyCapsule
        The capsule object of a dlpack tensor.

    Returns
    -------
    dragon.EagerTensor
        The tensor with the dlpack data.

    """
    ws = workspace.get_workspace()
    ref = EagerTensor(device=None)  # Hack the constructor.
    ref.__gc__ = ws.collectors.TENSOR
    ref._id = ref.__gc__.alloc('${DLPACK}')
    ref._impl = ws.CreateTensor(ref._id).FromDLPack(dlpack)
    ref._device = device_spec.DeviceSpec(*ref._impl.device)
    return ref


def to_dlpack(tensor, readonly=True):
    """Return a dlpack tensor sharing the data.

    Parameters
    ----------
    tensor : tensor_like
        The tensor to provide data.
    readonly : bool, optional, default=True
        **False** to sync the content with device.

    Returns
    -------
    PyCapsule
        The dlpack tensor object.

    """
    return tensor._impl.ToDLPack(tensor._device.to_proto(), readonly)
