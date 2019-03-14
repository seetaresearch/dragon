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
    def __init__(self, key, dev, **kwargs):
        super(Copy, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'Copy', 'arguments': {}}

    def forward(self, dst, src):
        outputs = [dst]; self.unify_devices(outputs)
        return self.run([src], outputs)


class Compare(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(Compare, self).__init__(key, dev, **kwargs)
        self.operation = kwargs.get('operation', 'NONE')
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Compare',
            'arguments': {
                'operation': self.operation,
                'to_uint8': True,
            }}

    def forward(self, x1, x2, y):
        inputs = [x1, x2]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output()]
        return self.run(inputs, outputs)