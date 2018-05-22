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


class AsType(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(AsType, self).__init__(key, ctx, **kwargs)
        self.dtype = kwargs.get('dtype', 'float32')
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for dtype op."""
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': 'AsType',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'dtype': self.dtype,
            }
        }

    def forward(self, x):
        if x.requires_grad and not x._static_shape:
            raise RuntimeError("Can't change the dtype of a non-leaf tensor if requiring grad.")
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(self.dtype)]
        y = self.run(inputs, outputs)
        y.requires_grad = x.requires_grad
        y._static_shape = x._static_shape
        return y