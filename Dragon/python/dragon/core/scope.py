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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core import tls as _tls


def name_scope(name):
    """Nest the specified name for naming tensors.

    Parameters
    ----------
    name : str
        The name adding to the tensors.

    Returns
    -------
    str
        The current nesting prefix.

    Examples
    --------
    >>> import dragon
    >>> with dragon.name_scope('conv1'): a = dragon.Tensor('weights')
    >>> a.name
    >>> "conv1/weights"

    """
    if name != '': prefix = name + _PREDEFINED_SCOPE_SEPARATOR
    else: prefix = '' # Avoid duplicated separators
    default = get_default_name_scope() + prefix
    return _GLOBAL_TENSOR_STACK.get_controller(default)


def device_scope(device_type, device_id=0):
    """Nest the the specific device info.

    Parameters
    ----------
    device_type : {'cpu', 'gpu', 'cuda', 'cnml'}, required
        The type of device.
    device_id : int, optional
        The index of the device.

    """
    device_type, device_id, device_type.lower(), device_id
    assert device_type in ('cpu', 'gpu', 'cuda', 'cnml')
    # Default names
    if device_type == 'gpu': device_type = 'cuda'
    return _GLOBAL_DEVICE_STACK.get_controller({
        'device_type': device_type,
            'device_id': device_id})


def phase_scope(phase):
    """Nest the the specific phase.

    Parameters
    ----------
    phase : {'TRAIN', 'TEST'}, required
        The phase.

    Returns
    -------
    str
        The specified phase.

    Examples
    --------
    >>> import dragon
    >>> a = dragon.ops.RandomUniform([2, 3])
    >>> with dragon.phase_scope(phase='TEST'): f_eval = dragon.function(outputs=a)

    """
    phase = phase.upper()
    assert phase in ('TRAIN', 'TEST'), \
        "Specified an unknown phase: " + phase
    return _GLOBAL_PHASE_STACK.get_controller(phase)


def get_default_name_scope():
    """Return the name scope in current nesting.

    Returns
    -------
    str
        The name scope.

    """
    ret = _GLOBAL_TENSOR_STACK.get_default()
    return ret if ret is not None else ''


def get_default_phase():
    """Return the phase in current nesting.

    Returns
    -------
    str or None
        The phase.

    """
    return _GLOBAL_PHASE_STACK.get_default()


def get_default_device():
    """Return the device dict in current nesting.

    The device dict contains the following keys:

    (``device_type``, ``device_id``).

    Returns
    -------
    dict
        The device dict.

    """
    return _GLOBAL_DEVICE_STACK.get_default()


_GLOBAL_TENSOR_STACK = _tls.Stack()
_GLOBAL_PHASE_STACK = _tls.Stack()
_GLOBAL_DEVICE_STACK = _tls.Stack()
_PREDEFINED_SCOPE_SEPARATOR = '/'