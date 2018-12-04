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

from dragon.import_c_apis import *

_TENSOR_SCOPE = ''
_PHASE_SCOPE = ''
_DEVICE_SCOPE = ''
_ENGINE_SCOPE = ''

SEPARATOR = '/'

_CURRENT_OP_UID = 0
_CURRENT_TENSOR_UID = 0

__all__ = [
    'GetTensorIdx',
    'GetTensorName',
    'GetOperatorIdx',
    'GetOperatorName',
    'TensorScope',
    'PhaseScope',
    'DeviceScope',
    'WorkspaceScope',
]

def GetOperatorIdx():
    """Get the available operator index.

    Returns
    -------
    int
        The operator index.

    """
    global _CURRENT_OP_UID
    _CURRENT_OP_UID += 1
    return _CURRENT_OP_UID - 1


def GetTensorIdx():
    """Get the available tensor index.

    Returns
    -------
    int
        The tensor index.

    """
    global _CURRENT_TENSOR_UID
    _CURRENT_TENSOR_UID += 1
    return _CURRENT_TENSOR_UID - 1


def GetOperatorName(name=None):
    """Get the available operator name.

    Parameters
    ----------
    name : str
        The optional name to use.

    Returns
    -------
    str
        The operator name.

    """
    op_idx = GetOperatorIdx()
    if name is None:
        return op_idx, 'Op_' + str(op_idx)
    else: return op_idx, name


def GetTensorName():
    """Get the available tensor name.

    Returns
    -------
    str
        The operator name.

    """
    return 'Tensor_' + str(GetTensorIdx())


class TensorScope(object):
    """TensorScope is the basic variable scope.

    Examples
    --------
    >>> import dragon as dg
    >>> with TensorScope('conv1'): a = dg.Tensor('weights')
    >>> a.name
    >>> "conv1/weights"

    >>> with dg.name_scope('conv2'): a = dg.Tensor('weights')
    >>> a.name
    >>> "conv2/weights"

    """
    def __init__(self, prefix):
        assert isinstance(prefix, type('str')), \
            "TensorScope takes in a string as its argument."
        if prefix != '':
            self.prefix = prefix + SEPARATOR
        else:
            # Avoid duplicated separators
            self.prefix = ''

    def __enter__(self):
        global _TENSOR_SCOPE
        _TENSOR_SCOPE += self.prefix
        return self.prefix.split(SEPARATOR)[0]

    def __exit__(self, type, value, traceback):
        global _TENSOR_SCOPE
        assert _TENSOR_SCOPE.endswith(self.prefix)
        if self.prefix != '':
            _TENSOR_SCOPE = _TENSOR_SCOPE[:-len(self.prefix)]


def get_tensor_scope():
    global _TENSOR_SCOPE
    return _TENSOR_SCOPE


def set_tensor_scope(name_scope):
    global _TENSOR_SCOPE
    _TENSOR_SCOPE = name_scope


class PhaseScope(object):
    """PhaseScope is a auxiliary to assign the specific phase.

    Examples
    --------
    >>> import dragon as dg
    >>> a = dg.ops.RandomUniform([2, 3])
    >>> with PhaseScope(phase='train'): f_train = dg.function(outputs=a)
    >>> with dg.phase_scope(phase='test'): f_eval = dg.function(outputs=a)

    """
    def __init__(self, phase):
        assert isinstance(phase, type('str')), \
            "PhaseScope takes in a string as its argument."
        self.phase = phase

    def __enter__(self):
        global _PHASE_SCOPE
        _PHASE_SCOPE = self.phase

    def __exit__(self, type, value, traceback):
        global _PHASE_SCOPE
        assert _PHASE_SCOPE == self.phase
        _PHASE_SCOPE = ''


class DeviceScope(object):
    """DeviceScope is a auxiliary to assign the specific device.

    Examples
    --------
    >>> import dragon as dg
    >>> with DeviceScope(device='cpu'): a = dg.ops.RandomUniform([2, 3])
    >>> with dg.device_scope(device='gpu', id=0, use_cudnn=True): b = dg.ops.RandomUniform([2, 3])

    """
    def __init__(self, device, id=0, use_cudnn=True):
        self.device = device.lower()
        self.engine = 'CUDNN' if use_cudnn else 'DRAGON'
        assert self.device in ['cpu', 'gpu', 'cuda']
        if self.device == 'cuda': self.device = 'gpu'
        self.id = id

    def __enter__(self):
        global _DEVICE_SCOPE, _ENGINE_SCOPE
        _DEVICE_SCOPE = '/' + self.device + ':' + str(self.id)
        _ENGINE_SCOPE = self.engine

    def __exit__(self, type, value, traceback):
        global _DEVICE_SCOPE, _ENGINE_SCOPE
        _DEVICE_SCOPE = ''
        _ENGINE_SCOPE = ''


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
        self.prev = CurrentWorkspaceCC()
        SwitchWorkspaceCC(self.ws, True)

    def __exit__(self, type, value, traceback):
        SwitchWorkspaceCC(self.prev, False)