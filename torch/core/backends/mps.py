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
"""MPS backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import backend
from dragon.core.framework import sysconfig
from dragon.core.framework import workspace


def current_device():
    """Return the index of current selected device.

    Returns
    -------
    int
        The device index.

    """
    return backend.mpsGetDevice()


def device_count():
    """Return the number of available devices.

    Returns
    -------
    int
        The number of devices.

    """
    return backend.mpsGetDeviceCount()


def get_device_family(device=None):
    """Return the supported families of specified device.

    If ``device`` is **None**, the current device will be selected.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int], optional
        The device to query.

    Returns
    -------
    List[str]
        The supported families.

    """
    device = -1 if device is None else device
    device_index = device.index if hasattr(device, 'index') else device
    return backend.mpsGetDeviceFamily(device_index)


def get_device_name(device=None):
    """Return the name of specified device.

    If ``device`` is **None**, the current device will be selected.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int], optional
        The device to query.

    Returns
    -------
    str
        The device name.

    """
    device = -1 if device is None else device
    device_index = device.index if hasattr(device, 'index') else device
    return backend.mpsGetDeviceName(device_index)


def is_available():
    """Return a bool reporting if MPS is available.

    Returns
    -------
    bool
        ``True`` if available otherwise ``False``.

    """
    return backend.mpsIsDriverSufficient()


def is_built():
    """Return a bool reporting if built with MPS support.

    Returns
    -------
    bool
        ``True`` if built otherwise ``False``.

    """
    return sysconfig.get_build_info()['is_mps_build']


def manual_seed(seed, device_index=None):
    """Set the random seed for mps device.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    seed : int
        The seed to use.
    device_index : int, optional
        The device index.

    """
    device_index = -1 if device_index is None else device_index
    backend.mpsSetRandomSeed(device_index, int(seed))


def manual_seed_all(seed):
    """Set the random seed for all mps devices.

    Parameters
    ----------
    seed : int
        The seed to use.

    """
    for i in range(backend.mpsGetDeviceCount()):
        backend.mpsSetRandomSeed(i, int(seed))


def memory_allocated(device=None):
    """Return the size of memory used by tensors in current workspace.

    If ``device`` is **None**, the current device will be selected.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int], optional
        The device to query.

    Returns
    -------
    int
        The total number of allocated bytes.

    """
    device_index = device.index if hasattr(device, 'index') else device
    if device_index is None:
        device_index = backend.mpsGetDevice()
    current_ws = workspace.get_workspace()
    return current_ws.memory_allocated('mps', device_index)


def set_device(device):
    """Set the current device.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int]
        The device to set.

    """
    device_index = device.index if hasattr(device, 'index') else device
    backend.mpsSetDevice(device_index)


def synchronize(device=None):
    """Synchronize all streams on a device.

    If ``device`` is **None**, the current device will be selected.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int], optional
        The device to synchronize.

    """
    device = -1 if device is None else device
    device_index = device.index if hasattr(device, 'index') else device
    backend.mpsStreamSynchronize(device_index, 0)
