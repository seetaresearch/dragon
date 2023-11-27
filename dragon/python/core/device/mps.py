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
"""MPS utilities."""

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
    return backend.mpsGetDevice()


def get_device_count():
    """Return the number of available devices.

    Returns
    -------
    int
        The number of devices.

    """
    return backend.mpsGetDeviceCount()


def get_device_family(device_index=None):
    """Return the supported families of specified device.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.

    Returns
    -------
    List[str]
        The supported families.

    """
    device_index = -1 if device_index is None else device_index
    return backend.mpsGetDeviceFamily(device_index)


def get_device_name(device_index=None):
    """Return the supported families of specified device.

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
    device_index = -1 if device_index is None else device_index
    return backend.mpsGetDeviceName(device_index)


def is_available():
    """Return a bool reporting if runtime is available.

    Returns
    -------
    bool
        ``True`` if available otherwise ``False``.

    """
    return backend.mpsIsDriverSufficient()


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
    return current_ws.memory_allocated("mps", device_index)


def set_default_device(device_index=0):
    """Set the default device.

    A valid device index should be greater equal than 0:

    ```python
    dragon.mps.set_default_device(0)   # Ok
    dragon.mps.set_default_device(-1)  # Reset to the cpu device
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
        config.config().device_type = "mps"
        config.config().device_index = device_index


def set_device(device_index=0):
    """Set the current device.

    Parameters
    ----------
    device_index : int, optional, default=0
        The device index.

    """
    backend.mpsSetDevice(device_index)


def set_random_seed(device_index=None, seed=3):
    """Set the random seed for mps device.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.
    seed : int, default=3
        The seed to use.

    """
    device_index = -1 if device_index is None else device_index
    backend.mpsSetRandomSeed(device_index, seed)


def synchronize(device_index=None, stream_index=0):
    """Synchronize a specified mps stream.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.
    stream_index : int, optional, default=0
        The stream index.

    """
    device_index = -1 if device_index is None else device_index
    backend.mpsStreamSynchronize(device_index, stream_index)
