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
"""Protocol buffer utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy

import google.protobuf.message as message_proto
import numpy

from dragon.core.framework import backend
from dragon.core.framework import config
from dragon.core.framework import context
from dragon.core.proto import dragon_pb2


def make_argument(key, value):
    """Make an argument."""
    argument = dragon_pb2.Argument()
    argument.name = key
    if type(value) is float:
        argument.f = value
    elif type(value) in (bool, int, numpy.int64):
        argument.i = value
    elif type(value) is bytes:
        argument.s = value
    elif isinstance(value, str):
        argument.s = str.encode(value)
    elif isinstance(value, message_proto.Message):
        argument.s = value.SerializeToString()
    elif all(type(v) is float for v in value):
        argument.floats.extend(value)
    elif all(type(v) is int for v in value):
        argument.ints.extend(value)
    elif all(type(v) is str for v in value):
        argument.strings.extend([str.encode(v) for v in value])
    elif all(type(v) is bytes for v in value):
        argument.strings.extend(value)
    elif all(isinstance(v, message_proto.Message) for v in value):
        argument.strings.extend([v.SerializeToString() for v in value])
    else:
        raise ValueError(
            'Unknown argument type: '
            'key = {}, value = {}, value_type = {}.'
            .format(key, value, type(value).__name__))
    return argument


def make_operator_def(
    op_type,
    inputs=(),
    outputs=(),
    name='',
    device_option=None,
    arg=None,
    cache_key=None,
    to_impl=False,
    **kwargs
):
    """Make an operator def."""
    op_def = dragon_pb2.OperatorDef(type=op_type, name=name)
    op_def.input.extend(inputs)
    op_def.output.extend(outputs)
    if device_option is not None:
        op_def.device_option.CopyFrom(device_option)
    if 'random_seed' in kwargs:
        op_def.device_option.random_seed = kwargs['random_seed']
        del kwargs['random_seed']
    if arg is not None:
        op_def.arg.extend(arg)
    for k, v in kwargs.items():
        if v is None:
            continue
        op_def.arg.add().CopyFrom(make_argument(k, v))
    if cache_key is not None:
        op_def.arg.add().CopyFrom(make_argument('cache_key', cache_key))
    if to_impl:
        impl = backend.OperatorDef()
        impl.ParseFrom(op_def.SerializeToString())
        return impl
    return op_def


def make_device_option(device_type, device_id, rng_seed=None):
    """Make a device option."""
    dev_opt = dragon_pb2.DeviceOption()
    dev_opt.device_type = device_type
    dev_opt.device_id = device_id
    if rng_seed is not None:
        dev_opt.random_seed = rng_seed
    return dev_opt


def get_device_option(device_type, device_index=0, rng_seed=None):
    """Return the device option."""
    option = _ALL_DEVICE_OPTIONS[(device_type, device_index)]
    if rng_seed is not None:
        option_copy = copy.deepcopy(option)
        option_copy.random_seed = rng_seed
        return option_copy
    return option


def get_default_device_option():
    """Return the default device option."""
    spec = context.get_device()
    if spec is not None:
        return get_device_option(spec.type, spec.index)
    return None


def get_global_device_option():
    """Return the global device option."""
    cfg = config.config()
    return get_device_option(cfg.device_type, cfg.device_index)


_MAX_NUM_OF_DEVICES = 16
_ALL_DEVICE_OPTIONS = {}
_DEVICE_TO_IDENTIFIER = {'cpu': 0, 'cuda': 1, 'mps': 2}

for i in range(_MAX_NUM_OF_DEVICES):
    for device, identifier in _DEVICE_TO_IDENTIFIER.items():
        _ALL_DEVICE_OPTIONS[(device, i)] = make_device_option(identifier, i)
