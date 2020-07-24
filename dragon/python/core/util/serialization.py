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
"""Serialization utility."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import cast
from typing import IO
from typing import Text


def save_bytes(str, f):
    """Save bytes to the file."""
    if hasattr(f, 'write') and callable(cast(IO[bytes], f).write):
        cast(IO[bytes], f).write(str)
    else:
        with open(cast(Text, f), 'wb') as writable:
            writable.write(str)


def load_bytes(f):
    """Load bytes from the file."""
    if hasattr(f, 'read') and callable(cast(IO[bytes], f).read):
        s = cast(IO[bytes], f).read()
    else:
        with open(cast(Text, f), 'rb') as readable:
            s = readable.read()
    return s


def serialize_proto(proto):
    """Serialize the protocol buffer object."""
    if proto is None:
        return b''
    elif isinstance(proto, bytes):
        return proto
    elif (hasattr(proto, 'SerializeToString') and
          callable(proto.SerializeToString)):
        result = proto.SerializeToString()
        return result
    else:
        raise ValueError(
            'No <SerializeToString> method. Type is {}'
            .format(type(proto)))


def deserialize_proto(s, proto):
    """Deserialize the protocol buffer object."""
    if not isinstance(s, bytes):
        raise ValueError('Excepted serialized bytes, got: {}'.format(type(s)))
    if not (hasattr(proto, 'ParseFromString') and
            callable(proto.ParseFromString)):
        raise ValueError(
            'No <ParseFromString> method. Type is {}'
            .format(type(proto)))
    proto.ParseFromString(s)
    return proto
