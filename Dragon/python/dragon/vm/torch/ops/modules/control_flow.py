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

from dragon.vm.torch.ops.modules.base import BaseModule


class Copy(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Copy, self).__init__(key, ctx, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Copy', 'arguments': {}}

    def forward(self, dst, src):
        outputs = [dst]; self.unify_devices(outputs)
        return self.run([src], outputs)