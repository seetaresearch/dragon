# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""Structure to represent a device."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import proto_util


class DeviceSpec(object):
    """Describe a computation device."""

    def __init__(self, type='cpu', index=0):
        """Create a ``DeviceSpec``."""
        self.type, self.index = type, index
        self._proto = None
        self._serialized_proto = None

    def copy(self):
        """Return a clone spec."""
        return DeviceSpec(self.type, self.index)

    def to_proto(self, serialized=True):
        """Return the device proto."""
        if self._proto is None:
            self._proto = proto_util.get_device_option(
                self.type, self.index)
        if serialized:
            if self._serialized_proto is None:
                self._serialized_proto = self._proto.SerializeToString()
            return self._serialized_proto
        return self._proto

    def __eq__(self, other):
        return self.type == other.type and self.index == other.index

    def __str__(self):
        return '{}:{}'.format(self.type, self.index)

    def __repr__(self):
        return 'device(type={}, index={})'.format(self.type, self.index)
