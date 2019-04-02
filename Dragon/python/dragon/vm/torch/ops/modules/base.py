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

import numpy

from dragon.core import proto_utils as _proto_utils
from dragon.core import workspace as _workspace
from dragon.vm.torch.module import Module


class BaseModule(Module):
    def __init__(self, key, dev, **kwargs):
        super(BaseModule, self).__init__()
        self._module_key = key
        self._device = dev
        self._args_dev = _proto_utils.\
            GetDeviceOption('cpu').SerializeToString()

    def set_argument_i64(self, name, value):
        _workspace.get_default_workspace()\
            .FeedTensor(name, numpy.array(
                value, dtype=numpy.int64), self._args_dev)