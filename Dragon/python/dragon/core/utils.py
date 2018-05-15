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

import sys
import numpy as np
from google.protobuf.message import Message

from dragon.protos import dragon_pb2 as pb


if sys.version_info >= (3,0):
    def MakeArgument(key, value):
        argument = pb.Argument()
        argument.name = key
        if type(value) is float: argument.f = value
        elif type(value) is int: argument.i = value
        elif type(value) is np.int64: argument.i64 = int(value)
        elif type(value) is str: argument.s = value
        elif type(value) is bool: argument.b = value
        elif isinstance(value, Message): argument.s = value.SerializeToString()
        elif all(type(v) is float for v in value): argument.floats.extend(value)
        elif all(type(v) is int for v in value): argument.ints.extend(value)
        elif all(type(v) is str for v in value): argument.strings.extend(value)
        elif all(isinstance(v, Message) for v in value):
            argument.strings.extend([v.SerializeToString() for v in value])
        else:
            raise ValueError('unknown argument type: key={} value={} value type={}' \
                             .format(key, value, type(value)))
        return argument
else:
    def MakeArgument(key, value):
        argument = pb.Argument()
        argument.name = key
        if type(value) is float: argument.f = value
        elif type(value) is int: argument.i = value
        elif type(value) is long: argument.i = value
        elif type(value) is np.int64: argument.i64 = int(value)
        elif type(value) is str: argument.s = value
        elif type(value) is unicode: argument.s = value
        elif type(value) is bool: argument.b = value
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
            raise ValueError('unknown argument type: key={} value={} value type={}' \
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


#  fix the python stdout
class Unbuffered(object):
   def __init__(self, stream):
       self.stream = stream
   def write(self, data):
       self.stream.write(data)
       self.stream.flush()
   def __getattr__(self, attr):
       return getattr(self.stream, attr)


# clear the stdout buffer for mpi(c++ & python)
import sys
sys.stdout = Unbuffered(sys.stdout)