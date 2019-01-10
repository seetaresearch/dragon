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

import dragon as dg
import numpy as np

import dragon.core.proto_utils as pb_utils
from dragon.vm.torch.module import Module


class BaseModule(Module):
    def __init__(self, key, ctx, **kwargs):
        super(BaseModule, self).__init__()
        self._persistent_key = key
        self._ctx = ctx
        self._args_dev = pb_utils.GetDeviceOption('CPU').SerializeToString()

    def register_argument(self, name):
        with dg.name_scope(self.persistent_key):
            return dg.Tensor(name).Variable().name

    def set_argument_i(self, name, value):
        dg.C.FeedTensorCC(name, np.array([value], dtype=np.int64), self._args_dev)