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
"""MLU utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import backend
from dragon.core.framework import config
from dragon.core.framework import workspace


def current_device():
    """Return the index of current selected device.

    Returns
    -------
    int
        The device index.

    """
    return backend.mluGetDevice()


def get_device_capability(device_index=None):
    """Return the capability of specified device.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.

    Returns
    -------
    Tuple[int, int]
        The major and minor number.

    """
    device_index = device_index if device_index else -1
    return backend.mluGetDeviceCapability(device_index)


def get_device_count():
    """Return the number of available devices.

    Returns
    -------
    int
        The number of devices.

    """
    return backend.mluGetDeviceCount()


def get_device_name(device_index=None):
    """Return the name of specified device.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.

    Returns
    -------
    str
        The device name.

    """
    device_index = device_index if device_index else -1
    return backend.mluGetDeviceName(device_index)


def is_available():
    """Return a bool reporting if runtime is available.

    Returns
    -------
    bool
        ``True`` if available otherwise ``False``.

    """
    return backend.mluIsDriverSufficient()


def memory_allocated(device_index=None):
    """Return the size of memory used by tensors in current workspace.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.

    Returns
    -------
    int
        The total number of allocated bytes.

    See Also
    --------
    `dragon.Workspace.memory_allocated(...)`_

    """
    if device_index is None:
        device_index = current_device()
    current_ws = workspace.get_workspace()
    return current_ws.memory_allocated("mlu", device_index)


def set_cnnl_flags(enabled=None):
    """Set the flags of CNNL library.

    Parameters
    ----------
    enabled : bool, optional
        Use CNNL library or not.

    """
    backend.cnnlSetFlags(-1 if enabled is None else enabled)


def set_default_device(device_index=0):
    """Set the default device.

    A valid device index should be greater equal than 0:

    ```python
    dragon.mlu.set_default_device(0)   # Ok
    dragon.mlu.set_default_device(-1)  # Reset to the cpu device
    ```

    Parameters
    ----------
    device_index : int
        The device index.

    """
    if device_index < 0:
        config.config().device_type = "cpu"
        config.config().device_index = 0
    else:
        config.config().device_type = "mlu"
        config.config().device_index = device_index


def set_device(device_index=0):
    """Set the current device.

    Parameters
    ----------
    device_index : int, optional, default=0
        The device index.

    """
    backend.mluSetDevice(device_index)


def set_random_seed(device_index=None, seed=3):
    """Set the random seed for mlu device.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.
    seed : int, default=3
        The seed to use.

    """
    device_index = -1 if device_index is None else device_index
    backend.mluSetRandomSeed(device_index, seed)


def synchronize(device_index=None, stream_index=0):
    """Synchronize a specified MLU stream.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.
    stream_index : int, optional, default=0
        The stream index.

    """
    device_index = -1 if device_index is None else device_index
    backend.mluStreamSynchronize(device_index, stream_index)
