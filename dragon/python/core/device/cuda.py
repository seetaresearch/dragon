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
"""CUDA utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import backend
from dragon.core.framework import config


class Stream(backend.CudaStream):
    """The CUDA stream wrapper."""

    def __init__(self, device_index):
        """Create a ``Stream``.

        Parameters
        ----------
        device_index : int, required
            The device index of stream.

        """
        super(Stream, self).__init__(device_index)

    @property
    def ptr(self):
        """Return the stream pointer.

        Returns
        -------
        int
            The address of pointer.

        """
        return super(Stream, self).ptr

    def synchronize(self):
        """Wait for the dispatched kernels to complete."""
        self.Synchronize()


def current_device():
    """Return the index of current selected device.

    Returns
    -------
    int
        The device index.

    """
    return backend.cudaGetDevice()


def enable_cudnn(enabled=True, benchmark=False):
    """Enable the CuDNN library.

    Parameters
    ----------
    enabled : bool, optional, default=True
        **True** to enable the CuDNN.
    benchmark : bool, optional, default=False
        **True** to select algorithms according to benchmark.

    """
    return backend.cudaEnableDNN(enabled, benchmark)


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
    return backend.cudaGetDeviceCapability(device_index)


def is_available():
    """Return a bool reporting if runtime is available.

    Returns
    -------
    bool
        **True** if available otherwise **False**.

    """
    return backend.cudaIsDriverSufficient()


def set_default_device(device_index=0):
    """Set the default device.

    A valid device index should be greater equal than 0:

    ```python
    dragon.cuda.set_default_device(0)   # Ok
    dragon.cuda.set_default_device(-1)  # Reset to the cpu device
    ```

    Parameters
    ----------
    device_index : int
        The device index.

    """
    if device_index < 0:
        config.config().device_type = 'cpu'
        config.config().device_index = 0
    else:
        config.config().device_type = 'cuda'
        config.config().device_index = device_index


def set_device(device_index=0):
    """Set the current device.

    Parameters
    ----------
    device_index : int, optional, default=0
        The device index.

    """
    return backend.cudaSetDevice(device_index)


def synchronize(device_index=None, stream_index=0):
    """Synchronize a specified CUDA stream.

    If ``device_index`` is **None**, the current device will be selected.

    Parameters
    ----------
    device_index : int, optional
        The device index.
    stream_index : int, optional, default=0
        The stream index.

    """
    device_index = device_index if device_index else -1
    return backend.cudaStreamSynchronize(device_index, stream_index)
