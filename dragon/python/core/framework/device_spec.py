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
"""Structure to represent a device."""

from dragon.core.framework import proto_util


class DeviceSpec(object):
    """Describe a computation device."""

    def __init__(self, type="cpu", index=0):
        """Create a ``DeviceSpec``.

        Parameters
        ----------
        type : str, optional, default='cpu'
            The device type.
        index : int, optional, default=0
            The device index.

        """
        self._type = type
        self._index = index
        self._proto = None
        self._serialized_proto = None

    @property
    def index(self):
        """Return the device index.

        Returns
        -------
        int
            The device index.

        """
        return self._index

    @property
    def type(self):
        """Return the device type.

        Returns
        -------
        str
            The device type.

        """
        return self._type

    def copy(self):
        """Return a cloned spec.

        Returns
        -------
        dragon.DeviceSpec
            The new device spec.

        """
        return DeviceSpec(self._type, self._index)

    def to_proto(self, serialized=True):
        """Return the device proto."""
        if self._proto is None:
            self._proto = proto_util.get_device_option(self._type, self._index)
        if serialized:
            if self._serialized_proto is None:
                self._serialized_proto = self._proto.SerializeToString()
            return self._serialized_proto
        return self._proto

    def __eq__(self, other):
        return self._type == other.type and self._index == other.index

    def __str__(self):
        return "{}:{}".format(self._type, self._index)

    def __repr__(self):
        return "device(type={}, index={})".format(self._type, self._index)
