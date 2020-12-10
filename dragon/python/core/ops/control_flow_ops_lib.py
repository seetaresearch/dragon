# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Control flow ops library."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework.ops import Operator


class Assign(Operator):
    """Assign operator."""

    def attributes(self):
        return {
            'op_type': 'Assign',
            'arguments': {
                'starts_desc': '${HANDLE}/starts',
                'sizes_desc': '${HANDLE}/sizes',
            },
        }

    def setup(self, ws, handle, starts, sizes):
        self.feed_arg(ws, '%s/starts' % handle, starts, 'int64')
        self.feed_arg(ws, '%s/sizes' % handle, sizes, 'int64')

    def forward(self, inputs, starts, sizes, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(
            inputs, outputs,
            callback=lambda ws, handle:
                self.setup(ws, handle, starts, sizes),
            no_grad=True,
        )


class MaskedAssign(Operator):
    """MaskedAssign operator."""

    def forward(self, inputs, inplace=False):
        outputs = [self.alloc(inputs[0]) if inplace else self.alloc()]
        return self.dispatch(inputs, outputs, no_grad=True)
