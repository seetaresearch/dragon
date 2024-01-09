# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Serialization utility."""

from typing import cast
from typing import IO
from typing import Text


def save_bytes(str, f):
    """Save bytes to the file."""
    if hasattr(f, "write") and callable(cast(IO[bytes], f).write):
        cast(IO[bytes], f).write(str)
    else:
        with open(cast(Text, f), "wb") as writable:
            writable.write(str)


def load_bytes(f):
    """Load bytes from the file."""
    if hasattr(f, "read") and callable(cast(IO[bytes], f).read):
        s = cast(IO[bytes], f).read()
    else:
        with open(cast(Text, f), "rb") as readable:
            s = readable.read()
    return s


def serialize_proto(proto):
    """Serialize the protocol buffer object."""
    if proto is None:
        return b""
    elif isinstance(proto, bytes):
        return proto
    elif hasattr(proto, "SerializeToString") and callable(proto.SerializeToString):
        result = proto.SerializeToString()
        return result
    else:
        raise ValueError("No <SerializeToString> method. Type is {}".format(type(proto)))


def deserialize_proto(s, proto):
    """Deserialize the protocol buffer object."""
    if not isinstance(s, bytes):
        raise ValueError("Excepted serialized bytes, got: {}".format(type(s)))
    if not (hasattr(proto, "ParseFromString") and callable(proto.ParseFromString)):
        raise ValueError("No <ParseFromString> method. Type is {}".format(type(proto)))
    proto.ParseFromString(s)
    return proto
