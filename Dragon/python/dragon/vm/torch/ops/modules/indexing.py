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


class Indexing(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Indexing, self).__init__(key, ctx, **kwargs)
        self.n_starts = kwargs.get('n_starts', 0)
        self.n_sizes = kwargs.get('n_sizes', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.starts = [self.register_argument(
             'starts[{}]'.format(i)) for i in range(self.n_starts)]
         self.sizes = [self.register_argument(
             'sizes[{}]'.format(i)) for i in range(self.n_sizes)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Crop',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'starts_desc': [e for e in self.starts] if self.n_starts > 0 else None,
                'sizes_desc': [e for e in self.sizes] if self.n_sizes > 0 else None,
            }
        }

    def forward(self, x, starts, sizes):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        for ix, d in enumerate(starts):
            self.set_argument_i(self.starts[ix], d)
        for ix, d in enumerate(sizes):
            self.set_argument_i(self.sizes[ix], d)
        return self.run(inputs, outputs)