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

from dragon.core.framework import backend
from dragon.core.framework import config
from dragon.core.framework import workspace


class Stream(backend.CUDAStream):
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
    return backend.cudaGetDeviceName(device_index)


def is_available():
    """Return a bool reporting if runtime is available.

    Returns
    -------
    bool
        ``True`` if available otherwise ``False``.

    """
    return backend.cudaIsDriverSufficient()


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
    return current_ws.memory_allocated('cuda', device_index)


def set_cublas_flags(allow_tf32=None):
    """Set the flags of cuBLAS library.

    Parameters
    ----------
    allow_tf32 : bool, optional, default=False
        Allow TF32 tensor core operation or not.

    """
    backend.cublasSetFlags(-1 if allow_tf32 is None else allow_tf32)


def set_cudnn_flags(
    enabled=None,
    benchmark=None,
    deterministic=None,
    allow_tf32=None,
):
    """Set the flags of cuDNN library.

    Parameters
    ----------
    enabled : bool, optional, default=True
        Use cuDNN library or not.
    benchmark : bool, optional, default=False
        Select fastest algorithms via benchmark or heuristics.
    deterministic : bool, optional, default=False
        Select deterministic algorithms instead of fastest.
    allow_tf32 : bool, optional, default=False
        Allow TF32 tensor core operation or not.

    """
    backend.cudnnSetFlags(
        -1 if enabled is None else enabled,
        -1 if benchmark is None else benchmark,
        -1 if deterministic is None else deterministic,
        -1 if allow_tf32 is None else allow_tf32)


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
    backend.cudaSetDevice(device_index)


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
    device_index = -1 if device_index is None else device_index
    backend.cudaStreamSynchronize(device_index, stream_index)
