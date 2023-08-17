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
"""CUDA memory utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import backend
from dragon.core.framework import workspace


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
    device_index = device.index if hasattr(device, "index") else device
    if device_index is None:
        device_index = backend.cudaGetDevice()
    current_ws = workspace.get_workspace()
    return current_ws.memory_allocated("cuda", device_index)
