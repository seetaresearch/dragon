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
"""Framework context."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import config
from dragon.core.framework import device_spec
from dragon.core.framework import mapping
from dragon.core.util import tls


def device(device_type, device_index=0):
    """Context-manager to nest the device.

    Examples:

    ```python
    with dragon.device('cuda', 0):
        x = dragon.constant(1)
    ```

    Parameters
    ----------
    device_type : str
        The device type.
    device_index : int, optional, default=0
        The device index.

    Returns
    -------
    dragon.DeviceSpec
        The current nesting device spec.

    """
    device_type = device_type.lower()
    if device_type not in mapping.DEVICE_STRING_TO_DEVICE_TYPE:
        raise ValueError('Unsupported device type: ' + device_type)
    device_type = mapping.DEVICE_STRING_TO_DEVICE_TYPE[device_type]
    spec = device_spec.DeviceSpec(device_type, device_index)
    return _GLOBAL_DEVICE_STACK.get_controller(spec)


def variable_scope(name):
    """Context-manager to nest the namespace for variables.

    Parameters
    ----------
    name : str
        The namespace for tensors traced by python.

    """
    return _GLOBAL_VARIABLE_SCOPE_STACK.get_controller(name)


def name_scope(name):
    """Context-manager to nest the name prefix for operations.

    Examples:

    ```python
    with dragon.name_scope('my_scope'):
        x = dragon.constant(1)
    print(x.name)
    ```

    Parameters
    ----------
    name : str
        The prefix name.

    Returns
    -------
    str
        The current nesting prefix.

    """
    if name != '':
        prefix = name + '/'
    else:
        prefix = ''  # Avoid duplicated separators.
    default = get_name_scope() + prefix
    return _GLOBAL_NAME_SCOPE_STACK.get_controller(default)


def get_device():
    """Return the nesting or default device."""
    spec = _GLOBAL_DEVICE_STACK.get_default()
    if spec is None:
        cfg = config.config()
        spec = device_spec.DeviceSpec(
            cfg.device_type, cfg.device_index)
    return spec


def get_variable_scope(persistent=False):
    """Return the variable scope in current nesting."""
    base = _GLOBAL_VARIABLE_SCOPE_STACK.get_default()
    return base + 'Ref' if persistent else base


def get_name_scope():
    """Return the name scope in current nesting."""
    ret = _GLOBAL_NAME_SCOPE_STACK.get_default()
    return ret if ret is not None else ''


# Thread-local stack for nesting scope.
_GLOBAL_DEVICE_STACK = tls.Stack()
_GLOBAL_VARIABLE_SCOPE_STACK = tls.Stack(['Variable'])
_GLOBAL_NAME_SCOPE_STACK = tls.Stack()
