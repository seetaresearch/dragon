# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""CUDA device utilities."""

from dragon.core.framework import backend
from dragon.core.framework import workspace


def current_device():
    """Return the index of current selected device.

    Returns
    -------
    int
        The device index.

    """
    return backend.cudaGetDevice()


def device_count():
    """Return the number of available devices.

    Returns
    -------
    int
        The number of devices.

    """
    return backend.cudaGetDeviceCount()


def get_device_capability(device=None):
    """Return the capability of specified device.

    If ``device`` is **None**, the current device will be selected.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int], optional
        The device to query.

    Returns
    -------
    Tuple[int, int]
        The major and minor number.

    """
    device = -1 if device is None else device
    device_index = device.index if hasattr(device, "index") else device
    return backend.cudaGetDeviceCapability(device_index)


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
    device_index = device.index if hasattr(device, "index") else device
    return backend.cudaGetDeviceName(device_index)


def is_available():
    """Return a bool reporting if runtime is available.

    Returns
    -------
    bool
        ``True`` if available otherwise ``False``.

    """
    return backend.cudaIsDriverSufficient()


def set_device(device):
    """Set the current device.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int]
        The device to set.

    """
    device_index = device.index if hasattr(device, "index") else device
    backend.cudaSetDevice(device_index)


def synchronize(device=None):
    """Synchronize the workspace stream on a device.

    If ``device`` is **None**, the current device will be selected.

    Parameters
    ----------
    device : Union[dragon.vm.torch.device, int], optional
        The device to synchronize.

    """
    device = -1 if device is None else device
    device_index = device.index if hasattr(device, "index") else device
    stream_index = workspace.get_workspace().get_stream()
    backend.cudaStreamSynchronize(device_index, stream_index)
