# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from collections import defaultdict

TENSOR_SCOPE = ''
PHASE_SCOPE = ''
DEVICE_SCOPE = ''
ENGINE_SCOPE = ''

_CURRENT_OP_IDX = 0
_SCOPE_TENSOR_IDX = defaultdict(int)

def GetOperatorIdx():
    global _CURRENT_OP_IDX
    _CURRENT_OP_IDX = _CURRENT_OP_IDX + 1
    return _CURRENT_OP_IDX - 1

def GetTensorIdx():
    global _SCOPE_TENSOR_IDX
    _SCOPE_TENSOR_IDX[TENSOR_SCOPE] += 1
    return _SCOPE_TENSOR_IDX[TENSOR_SCOPE] - 1

def GetOperatorName(name=None):
    op_idx = GetOperatorIdx()
    if name is None:
        return op_idx, 'Op_' + str(op_idx)
    else: return op_idx, name

def GetTensorName():
    return 'Tensor_' + str(GetTensorIdx())

class TensorScope(object):
    SEPARATOR = '/'
    def __init__(self, prefix):
        assert isinstance(prefix, basestring), \
            "TensorScope takes in a string as its argument."
        self.prefix = prefix + TensorScope.SEPARATOR

    def __enter__(self):
        global TENSOR_SCOPE
        TENSOR_SCOPE += self.prefix

    def __exit__(self, type, value, traceback):
        global TENSOR_SCOPE
        assert TENSOR_SCOPE.endswith(self.prefix)
        TENSOR_SCOPE = TENSOR_SCOPE[:-len(self.prefix)]

class PhaseScope(object):
    def __init__(self, phase):
        assert isinstance(phase, basestring), \
            "PhaseScope takes in a string as its argument."
        self.phase = phase

    def __enter__(self):
        global PHASE_SCOPE
        PHASE_SCOPE = self.phase

    def __exit__(self, type, value, traceback):
        global PHASE_SCOPE
        assert PHASE_SCOPE == self.phase
        PHASE_SCOPE = ''

class DeviceScope(object):
    def __init__(self, device, id=0, use_cudnn=True):
        self.device = device.lower()
        self.engine = 'CUDNN' if use_cudnn else 'DRAGON'
        assert self.device in ['cpu', 'gpu', 'cuda']
        if self.device == 'cuda': self.device = 'gpu'
        self.id = id

    def __enter__(self):
        global DEVICE_SCOPE, ENGINE_SCOPE
        DEVICE_SCOPE = '/' + self.device + ':' + str(self.id)
        ENGINE_SCOPE = self.engine


    def __exit__(self, type, value, traceback):
        global DEVICE_SCOPE, ENGINE_SCOPE
        DEVICE_SCOPE = ''
        ENGINE_SCOPE = ''
