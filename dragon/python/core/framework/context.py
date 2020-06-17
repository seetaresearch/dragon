# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import config
from dragon.core.framework import device_spec
from dragon.core.util import tls


def device(device_type, device_id=0):
    """Context-manager to nest the the device spec.

    Examples:

    ```python
    with dragon.device('cuda', 0):
        x = dragon.constant(1)
    ```

    Parameters
    ----------
    device_type : {'cpu', 'gpu', 'cuda', 'cnml'}, required
        The type of device.
    device_id : int, optional, default=0
        The index of the device.

    Returns
    -------
    Dict
        The current default device spec.

    """
    device_type, device_id, device_type.lower(), device_id
    assert device_type in ('cpu', 'gpu', 'cuda', 'cnml')
    if device_type == 'gpu':
        device_type = 'cuda'
    return _GLOBAL_DEVICE_STACK.get_controller({
        'device_type': device_type,
        'device_index': device_id,
    })


def eager_scope(data='${DATA}', graph='${GRAPH}'):
    """Context-manager to nest the domain for eager resources.

    Parameters
    ----------
    data : str, optional, default='${DATA}'
        The domain for resources traced by python.
    graph : str, optional, default='${GRAPH}'
        The domain for resources traced by graph.

    """
    domain_tuple = (graph, data)
    return _GLOBAL_EAGER_STACK.get_controller(domain_tuple)


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
    return _GLOBAL_NAME_STACK.get_controller(default)


def graph_phase(phase):
    """Context-manager to nest the the executing phase for graph.

    Parameters
    ----------
    phase : {'TRAIN', 'TEST'}, required
        The executing phase.

    """
    phase = phase.upper()
    assert phase in ('TRAIN', 'TEST'), \
        "Specified an unknown phase: " + phase
    return _GLOBAL_PHASE_STACK.get_controller(phase)


def get_device_info():
    """Return the device info in current nesting."""
    return _GLOBAL_DEVICE_STACK.get_default()


def get_device_spec():
    """Return the device spec in current nesting."""
    dev_info = get_device_info()
    if dev_info is not None:
        return device_spec.DeviceSpec(
            dev_info['device_type'],
            dev_info['device_index'],
        )
    else:
        cfg = config.config()
        return device_spec.DeviceSpec(
            cfg.device_type,
            cfg.device_index,
        )


def get_eager_scope(requires_grad=False):
    """Return the eager scope in current nesting."""
    ret = _GLOBAL_EAGER_STACK.get_default()
    return ret[0] if requires_grad else ret[1]


def get_name_scope():
    """Return the name scope in current nesting."""
    ret = _GLOBAL_NAME_STACK.get_default()
    return ret if ret is not None else ''


def get_graph_phase():
    """Return the graph phase in current nesting."""
    return _GLOBAL_PHASE_STACK.get_default()


# Thread-local stack for nesting scope.
_GLOBAL_DEVICE_STACK = tls.Stack()
_GLOBAL_EAGER_STACK = tls.Stack([('${GRAPH}', '${DATA}')])
_GLOBAL_NAME_STACK = tls.Stack()
_GLOBAL_PHASE_STACK = tls.Stack()
