# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/onnx/onnx/blob/master/onnx/__init__.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import cast
from typing import IO
from typing import Optional
from typing import Text

from google.protobuf import message
try:
    from onnx import ModelProto
except ImportError:
    from dragon.core.util import deprecation
    ModelProto = deprecation.not_installed('onnx')


# str should be bytes,
# f should be either writable or a file path.
def _save_bytes(str, f):
    if hasattr(f, 'write') and \
            callable(cast(IO[bytes], f).write):
        cast(IO[bytes], f).write(str)
    else:
        with open(cast(Text, f), 'wb') as writable:
            writable.write(str)


# f should be either readable or a file path.
def _load_bytes(f):
    if hasattr(f, 'read') and \
            callable(cast(IO[bytes], f).read):
        s = cast(IO[bytes], f).read()
    else:
        with open(cast(Text, f), 'rb') as readable:
            s = readable.read()
    return s


def _serialize(proto):
    if isinstance(proto, bytes):
        return proto
    elif hasattr(proto, 'SerializeToString') and \
            callable(proto.SerializeToString):
        result = proto.SerializeToString()
        return result
    else:
        raise ValueError(
            'No SerializeToString method is detected. '
            'neither proto is a str.\ntype is {}'
            .format(type(proto))
        )


def _deserialize(s, proto):
    if not isinstance(s, bytes):
        raise ValueError(
            'Parameter s must be bytes, '
            'but got type: {}'
            .format(type(s))
        )

    if not (hasattr(proto, 'ParseFromString') and
            callable(proto.ParseFromString)):
        raise ValueError(
            'No ParseFromString method is detected. '
            '\ntype is {}'.format(type(proto))
        )

    decoded = cast(Optional[int], proto.ParseFromString(s))
    if decoded is not None and decoded != len(s):
        raise message.DecodeError(
            "Protobuf decoding consumed too few bytes: {} out of {}"
            .format(decoded, len(s))
        )
    return proto


def save_model(proto, f):
    s = _serialize(proto)
    _save_bytes(s, f)


def load_model_from_string(s):
    if ModelProto is None:
        raise ImportError('ONNX is not installed.')
    return _deserialize(s, ModelProto())


def load_model(f):
    s = _load_bytes(f)
    return load_model_from_string(s)


load = load_model
load_from_string = load_model_from_string
save = save_model
