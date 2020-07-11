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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Assign(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Assign, self).__init__(key, dev, **kwargs)
        self.ndim = kwargs.get('ndim', 0)

    def attributes(self):
        return {
            'op_type': 'Assign',
            'arguments': {
                'starts_descs': [
                    '${{HANDLE}}/starts[{}]'
                    .format(n) for n in range(self.ndim)],
                'sizes_descs': [
                    '${{HANDLE}}/sizes[{}]'
                    .format(n) for n in range(self.ndim)],
            },
        }

    def feed(self, ws, handle, starts, sizes):
        for i in range(self.ndim):
            self.feed_arg(
                ws, '{}/starts[{}]'.format(handle, i),
                starts[i], 'int64')
            self.feed_arg(
                ws, '{}/sizes[{}]'.format(handle, i),
                sizes[i], 'int64')

    def forward(self, inputs, starts, sizes):
        return self.dispatch(
            [inputs[1]], [inputs[0]],
            callback=lambda ws, handle:
                self.feed(ws, handle, starts, sizes),
            no_grad=True,
        )


class Copy(Operator):
    def __init__(self, key, dev, **kwargs):
        super(Copy, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {'op_type': 'Copy', 'arguments': {}}

    def forward(self, inputs, outputs):
        outputs = outputs if outputs else [self.alloc()]
        return self.dispatch(inputs, outputs)


class MaskedAssign(Operator):
    def __init__(self, key, dev, **kwargs):
        super(MaskedAssign, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {'op_type': 'MaskedAssign', 'arguments': {}}

    def forward(self, inputs):
        return self.dispatch(inputs[1:], [inputs[0]], no_grad=True)
