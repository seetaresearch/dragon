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

import threading
import dragon.import_c_api as C

from contextlib import contextmanager


__all__ = [
    'name_scope',
    'phase_scope',
    'device_scope',
    'get_default_phase',
    'get_default_device',
    'get_default_name_scope',
    'WorkspaceScope',
]


class _ThreadLocalStack(threading.local):
    def __init__(self):
        super(_ThreadLocalStack, self).__init__()
        self._enforce_nesting = True
        self.stack = []

    def get_default(self):
        return self.stack[-1] if len(self.stack) >= 1 else None

    def is_cleared(self):
        return not self.stack

    @property
    def enforce_nesting(self):
        return self._enforce_nesting

    @enforce_nesting.setter
    def enforce_nesting(self, value):
        self._enforce_nesting = value

    @contextmanager
    def get_controller(self, default):
        """A context manager for manipulating a default stack."""
        self.stack.append(default)
        try:
            yield default
        finally:
            # stack may be empty if reset() was called
            if self.stack:
                if self._enforce_nesting:
                    if self.stack[-1] is not default:
                        raise AssertionError(
                            "Nesting violated for default stack of %s objects" %
                            type(default))
                    self.stack.pop()
                else:
                    self.stack.remove(default)


class WorkspaceScope(object):
    """WorkspaceScope is a auxiliary to assign the specific workspace.

    Examples
    --------
    >>> import dragon as dg
    >>> with WorkspaceScope('session1'): pass
    >>> with dg.workspace_scope('session2'): pass

    """
    def __init__(self, ws_name):
        assert isinstance(ws_name, type('str')), \
            'WorkspaceScope takes in a string as its argument.'
        assert ws_name != '', \
            'The workspace name should not be empty.'
        self.ws = ws_name
        self.prev = 'default'

    def __enter__(self):
        self.prev = C.CurrentWorkspace()
        C.SwitchWorkspace(self.ws, True)

    def __exit__(self, type, value, traceback):
        C.SwitchWorkspace(self.prev, True)


_GLOBAL_TENSOR_STACK = _ThreadLocalStack()
_GLOBAL_PHASE_STACK = _ThreadLocalStack()
_GLOBAL_DEVICE_STACK = _ThreadLocalStack()
_PREDEFINED_SCOPE_SEPARATOR = '/'


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


def device_scope(device_type, device_id=0, engine='AUTO'):
    """Nest the the specific device info.

    Parameters
    ----------
    device_type : {'CPU', 'GPU', 'CUDA', 'CNML'}, required
        The type of device.
    device_id : int, optional
        The index of the device.
    engine : {'AUTO', 'CUDNN'}, optional
        The auxiliary accelerating library to use.

    """
    device_type, device_id, device_engine = \
        device_type.upper(), device_id, engine.upper()
    assert device_type in ['CPU', 'GPU', 'CUDA', 'CNML']
    # Default names
    if device_type == 'GPU': device_type = 'CUDA'
    if device_engine == 'AUTO': device_engine = 'CUDNN'
    return _GLOBAL_DEVICE_STACK.get_controller({
        'device_type': device_type,
        'device_id': device_id,
        'device_engine': device_engine})


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

    (``device_type``, ``device_id``, ``device_engine``).

    Returns
    -------
    dict
        The device dict.

    """
    return _GLOBAL_DEVICE_STACK.get_default()