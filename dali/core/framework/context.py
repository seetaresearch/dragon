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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core import distributed
from dragon.core.util import tls


def device(device_type, device_index=0):
    """Context-manager to nest the device spec.

    Examples:

    ```python
    with dali.device('cuda', 0):
        pass

    ```
    Parameters
    ----------
    device_type : {'cpu', 'gpu', 'cuda'}, required
        The type of device.
    device_index : int, optional, default=0
        The index of the device.


    """
    device_type = device_type.lower()
    assert device_type in ('cpu', 'gpu', 'cuda')
    if device_type == 'gpu':
        device_type = 'cuda'
    return _GLOBAL_DEVICE_STACK.get_controller({
        'device_type': device_type,
        'device_index': device_index,
    })


def get_device():
    """Return the device dict in current nesting."""
    return _GLOBAL_DEVICE_STACK.get_default()


def get_device_type(mixed=False):
    """Return the current nesting device type.

    Parameters
    ----------
    mixed : bool, optional, default=False
        ``True`` to return ``mixed`` for gpu device.

    Returns
    -------
    {'cpu', 'gpu', 'mixed'}
        The current device type.

    """
    device_type = get_device()['device_type']
    if device_type == 'cuda':
        return 'mixed' if mixed else 'gpu'
    else:
        return 'cpu'


def get_distributed_info(allowed=True):
    """Return the rank and size of current nesting group.

    Parameters
    ----------
    allowed : bool, optional, default=True
        Whether the distributed utilities are allowed.

    Returns
    -------
    Tuple[int]
        The node rank and group size.

    """
    if allowed:
        group = distributed.get_group()
        if group is not None:
            return distributed.get_rank(group), group.size
    return 0, 1


_GLOBAL_DEVICE_STACK = tls.Stack(
    [{'device_type': 'cpu', 'device_index': 0}]
)
