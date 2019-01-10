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

"""Define some helpful protobuf makers here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import numpy as np
from google.protobuf.message import Message

import dragon.config as cfg
from dragon.proto import dragon_pb2 as pb
from dragon.core.scope import get_default_device


if sys.version_info >= (3,0):
    def MakeArgument(key, value):
        argument = pb.Argument()
        argument.name = key
        if type(value) is float: argument.f = value
        elif type(value) in (bool, int, np.int64) : argument.i = value
        elif type(value) is bytes: argument.s = value
        elif type(value) is str: argument.s = str.encode(value)
        elif isinstance(value, Message): argument.s = value.SerializeToString()
        elif all(type(v) is float for v in value): argument.floats.extend(value)
        elif all(type(v) is int for v in value): argument.ints.extend(value)
        elif all(type(v) is str for v in value): argument.strings.extend([str.encode(v) for v in value])
        elif all(isinstance(v, Message) for v in value):
            argument.strings.extend([v.SerializeToString() for v in value])
        else:
            raise ValueError(
                'Unknown argument type: key = {}, value = {}, value_type = {}.' \
                    .format(key, value, type(value)))
        return argument
else:
    def MakeArgument(key, value):
        argument = pb.Argument()
        argument.name = key
        if type(value) is float: argument.f = value
        elif type(value) in (bool, int, long, np.int64) : argument.i = value
        elif type(value) in (str, unicode): argument.s = value
        elif isinstance(value, Message): argument.s = value.SerializeToString()
        elif all(type(v) is float for v in value): argument.floats.extend(value)
        elif all(type(v) is int for v in value): argument.ints.extend(value)
        elif all(type(v) is long for v in value): argument.ints.extend(value)
        elif all(type(v) is str for v in value): argument.strings.extend(value)
        elif all(type(v) is unicode or type(v) is str for v in value):
            argument.strings.extend(value)
        elif all(isinstance(v, Message) for v in value):
            argument.strings.extend([v.SerializeToString() for v in value])
        else:
            raise ValueError(
                'Unknown argument type: key = {}, value = {}, value_type = {}.' \
                    .format(key, value, type(value)))
        return argument


def MakeOperatorDef(op_type, inputs, outputs, name='',
                    device_option=None, arg=None, engine=None, **kwargs):
    operator = pb.OperatorDef()
    operator.type = op_type
    operator.name = name
    operator.input.extend([str(tensor) for tensor in inputs])
    operator.output.extend([str(tensor) for tensor in outputs])
    if device_option is not None:
        operator.device_option.CopyFrom(device_option)
        if engine is not None:
            operator.device_option.engine = engine
    if 'random_seed' in kwargs:
        operator.device_option.random_seed = kwargs['random_seed']
        del kwargs['random_seed']
    if arg is not None:
        operator.arg.extend(arg)
    for k,v in kwargs.items():
        if v is None: continue
        operator.arg.add().CopyFrom(MakeArgument(k,v))
    return operator


def MutableOperatorDef(meta_def, inputs, outputs):
    op = pb.OperatorDef(); op.CopyFrom(meta_def)
    op.ClearField('input'); op.input.extend(inputs)
    op.ClearField('output'); op.output.extend(outputs)
    return op


def MakeDeviceOption(device_type, device_id, engine=None, rng_seed=None):
    option = pb.DeviceOption()
    option.device_type = device_type
    option.device_id = device_id
    if engine is not None: option.engine = engine
    if rng_seed is not None: option.random_seed = rng_seed
    return option


_PREDEFINED_DEVICE_LIMITS = 16
_PREDEFINED_DEVICE_ENGINES = ['', 'CUDNN']
_PREDEFINED_DEVICE_DICT = {'CPU': 0, 'CUDA': 1, 'CNML': 2}
_PREDEFINED_DEVICE_OPTION_DICT = {}


for i in range(_PREDEFINED_DEVICE_LIMITS):
    for device, identify in _PREDEFINED_DEVICE_DICT.items():
        for engine in _PREDEFINED_DEVICE_ENGINES:
            _PREDEFINED_DEVICE_OPTION_DICT[(device, i, engine)] = \
                MakeDeviceOption(identify, i, engine)
        if device == 'CUDA':
            _PREDEFINED_DEVICE_OPTION_DICT[('CUDA', i)] = \
                MakeDeviceOption(identify, i, 'CUDNN')


def GetDeviceOption(device_type, device_id=0, engine=None, rng_seed=None):
    ctx = (device_type, device_id, engine if engine else '')
    option = _PREDEFINED_DEVICE_OPTION_DICT[ctx]
    if rng_seed is not None:
        option_copy = copy.deepcopy(option)
        option_copy.random_seed = rng_seed
        return option_copy
    return option


def GetDefaultDeviceOption():
    device_info = get_default_device()
    if device_info is not None:
        return GetDeviceOption(
            device_info['device_type'],
            device_info['device_id'],
            device_info['device_engine'])
    return None


def GetGlobalDeviceOption():
    option = cfg.GetGlobalOptions()
    return GetDeviceOption(
        option['device'], option['device_id'],
            'CUDNN' if option['use_cudnn'] else '')


# Fix the python stdout
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


# Clear the stdout buffer for mpi(C++ && Python)
import sys
sys.stdout = Unbuffered(sys.stdout)