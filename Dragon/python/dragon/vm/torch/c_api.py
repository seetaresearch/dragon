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

import copy


class Size(tuple):
    def __init__(self, sizes):
        super(Size, self).__init__()

    def __setitem__(self, key, value):
        raise TypeError("'torch.Size' object does not support item assignment")

    def __repr__(self):
        return 'torch.Size([{}])'.format(', '.join([str(s) for s in self]))


class Context(object):
    def __init__(self, device_type='CPU', device_id=0):
        self._device_type = device_type
        self._device_id = device_id

    @property
    def device_type(self):
        return self._device_type

    @device_type.setter
    def device_type(self, value):
        self._device_type = value

    @property
    def device_id(self):
        return self._device_id

    @device_id.setter
    def device_id(self, value):
        self._device_id = value

    def copy(self):
        return copy.deepcopy(self)

    def __str__(self):
        return '{}:{}'.format(
            self._device_type, self._device_id)