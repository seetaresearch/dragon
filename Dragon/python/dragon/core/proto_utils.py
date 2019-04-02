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

"""Define some helpful protocol buffer makers here."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import copy
import numpy

from dragon import config as _cfg
from dragon import import_c_api as _C
from dragon.core import scope as _scope
from dragon.proto import dragon_pb2 as _proto_def
from google.protobuf.message import Message as _Message


if sys.version_info >= (3,0):
    def MakeArgument(key, value):
        argument = _proto_def.Argument()
        argument.name = key
        if type(value) is float: argument.f = value
        elif type(value) in (bool, int, numpy.int64) : argument.i = value
        elif type(value) is bytes: argument.s = value
        elif type(value) is str: argument.s = str.encode(value)
        elif isinstance(value, _Message): argument.s = value.SerializeToString()
        elif all(type(v) is float for v in value): argument.floats.extend(value)
        elif all(type(v) is int for v in value): argument.ints.extend(value)
        elif all(type(v) is str for v in value): argument.strings.extend([str.encode(v) for v in value])
        elif all(isinstance(v, _Message) for v in value):
            argument.strings.extend([v.SerializeToString() for v in value])
        else:
            raise ValueError(
                'Unknown argument type: key = {}, value = {}, value_type = {}.' \
                    .format(key, value, type(value)))
        return argument
else:
    def MakeArgument(key, value):
        argument = _proto_def.Argument()
        argument.name = key
        if type(value) is float: argument.f = value
        elif type(value) in (bool, int, long, numpy.int64) : argument.i = value
        elif type(value) is str: argument.s = value
        elif type(value) is unicode: argument.s = str(value)
        elif isinstance(value, _Message): argument.s = value.SerializeToString()
        elif all(type(v) is float for v in value): argument.floats.extend(value)
        elif all(type(v) is int for v in value): argument.ints.extend(value)
        elif all(type(v) is long for v in value): argument.ints.extend(value)
        elif all(type(v) is str for v in value): argument.strings.extend(value)
        elif all(type(v) is unicode for v in value):
            argument.strings.extend([str(v) for v in value])
        elif all(isinstance(v, _Message) for v in value):
            argument.strings.extend([v.SerializeToString() for v in value])
        else:
            raise ValueError(
                'Unknown argument type: key = {}, value = {}, value_type = {}.' \
                    .format(key, value, type(value)))
        return argument


def MakeOperatorDef(
    op_type,
    inputs=(),
    outputs=(),
    name='',
    uid=None,
    device_option=None,
    arg=None,
    **kwargs
):
    operator = _proto_def.OperatorDef()
    operator.type = op_type
    operator.name = name
    operator.input.extend([str(tensor) for tensor in inputs])
    operator.output.extend([str(tensor) for tensor in outputs])
    if device_option is not None:
        operator.device_option.CopyFrom(device_option)
    if 'random_seed' in kwargs:
        operator.device_option.random_seed = kwargs['random_seed']
        del kwargs['random_seed']
    if uid is not None: operator.uid = uid
    if arg is not None: operator.arg.extend(arg)
    for k, v in kwargs.items():
        if v is None: continue
        operator.arg.add().CopyFrom(MakeArgument(k,v))
    return operator


def MakeCXXOperatorDef(
    op_type,
    inputs=(),
    outputs=(),
    name='',
    uid=None,
    device_option=None,
    arg=None,
    **kwargs
):
    c_def = _C.OperatorDef()
    py_def = MakeOperatorDef(
        op_type, inputs, outputs, name, uid,
            device_option, arg, **kwargs)
    c_def.ParseFrom(py_def.SerializeToString())
    return c_def


def MakeDeviceOption(device_type, device_id, rng_seed=None):
    option = _proto_def.DeviceOption()
    option.device_type = device_type
    option.device_id = device_id
    if rng_seed is not None: option.random_seed = rng_seed
    return option


_PREDEFINED_DEVICE_LIMITS = 16
_PREDEFINED_DEVICE_DICT = {'cpu': 0, 'cuda': 1, 'cnml': 2}
_PREDEFINED_DEVICE_OPTION_DICT = {}


for i in range(_PREDEFINED_DEVICE_LIMITS):
    for device, identify in _PREDEFINED_DEVICE_DICT.items():
        _PREDEFINED_DEVICE_OPTION_DICT[(device, i)] = \
            MakeDeviceOption(identify, i)


def GetDeviceOption(device_type, device_id=0, rng_seed=None):
    ctx = (device_type, device_id)
    option = _PREDEFINED_DEVICE_OPTION_DICT[ctx]
    if rng_seed is not None:
        option_copy = copy.deepcopy(option)
        option_copy.random_seed = rng_seed
        return option_copy
    return option


def GetDefaultDeviceOption():
    device_info = _scope.get_default_device()
    if device_info is not None:
        return GetDeviceOption(
            device_info['device_type'],
                device_info['device_id'])
    return None


def GetGlobalDeviceOption():
    options = _cfg.GetGlobalOptions()
    return GetDeviceOption(
        options['device'],
            options['device_id'])


# Fix the python stdout
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


# Clear the stdout buffer for mpi
sys.stdout = Unbuffered(sys.stdout)