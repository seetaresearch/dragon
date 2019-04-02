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

"""List some useful CUDA C++ API."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import import_c_api as _C


def IsCUDADriverSufficient():
    """Is cuda driver sufficient?

    Returns
    -------
    boolean
        *True* if your device(s) support CUDA otherwise *False*.

    """
    return _C.IsCUDADriverSufficient()


def EnableCUDNN(enabled=True):
    """Enable the CuDNN engine.

    Parameters
    ----------
    enabled : boolean
        *True* to enable.

    Returns
    -------
    None

    """
    return _C.EnableCUDNN(enabled)


def GetDevice():
    """Get the current active cuda device.

    Returns
    -------
    int
        The device index.

    """
    return _C.cudaGetDevice()


def SynchronizeStream(device_id=None, stream_id=0):
    """Synchronize the specified cuda stream.

    If ``device_id`` is *None*, the current active device will be selected.

    Returns
    -------
    device_id : int or None
        The device index.
    stream_id : int
        The stream index.

    """
    return _C.cudaStreamSynchronize(
        device_id if device_id else -1, stream_id)