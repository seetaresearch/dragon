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


class Reduce(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Reduce, self).__init__(key, ctx, **kwargs)
        self.op_type = kwargs.get('op_type', 'Reduce')
        self.tag = kwargs.get('tag', None)
        self.axis = kwargs.get('axis', -1)
        self.keep_dims = kwargs.get('keep_dims', True)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
        """No Arguments for reduce op.

        Mutable ``axis`` and ``keep_dims`` is non-trivial for backend,
        we simply hash them in the persistent key.

        """
        pass

    def register_op(self):
        self.op_meta = {
            'op_type': self.op_type,
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'operation': self.tag,
                'axis': self.axis,
                'keep_dims': self.keep_dims
            }
        }

    def forward(self, x, y):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [y] if y else [self.register_output(x.dtype)]
        return self.run(inputs, outputs)