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


class Assign(BaseModule):
    """This module imports the *AssignOp* from backend.

    Arbitrary length of starts and sizes will be take.

    """
    def __init__(self, key, dev, **kwargs):
        super(Assign, self).__init__(key, dev, **kwargs)
        self.nstarts = kwargs.get('nstarts', 0)
        self.nsizes = kwargs.get('nsizes', 0)
        self.register_op()

    def register_op(self):
        self.op_meta = {
            'op_type': 'Assign',
            'arguments': {
                'starts_desc': [
                    '${{HANDLE}}/starts[{}]'.format(n)
                        for n in range(self.nstarts)],
                'sizes_desc': [
                    '${{HANDLE}}/sizes[{}]'.format(n)
                        for n in range(self.nsizes)],
            },
        }

    def update_args(self, A, starts, sizes):
        for i, e in enumerate(starts):
            self.set_arg_i64('{}/starts[{}]'.format(A, i), e)
            self.set_arg_i64('{}/sizes[{}]'.format(A, i), sizes[i])

    def forward(self, x, y, starts, sizes):
        self.unify_devices([x, y])
        callback = lambda A: self.update_args(A, starts, sizes)
        return self.run([x], [y], callback=callback, auto_grad=False)


class MaskedAssign(BaseModule):
    def __init__(self, key, dev, **kwargs):
        super(MaskedAssign, self).__init__(key, dev, **kwargs)
        self.register_op()

    def register_op(self):
        self.op_meta = {'op_type': 'MaskedAssign', 'arguments': {}}

    def forward(self, x, y, mask):
        self.unify_devices([x, y])
        return self.run([x, mask], [y])