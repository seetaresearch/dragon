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


class Crop(BaseModule):
    def __init__(self, key, ctx, **kwargs):
        super(Crop, self).__init__(key, ctx, **kwargs)
        self.len_starts = kwargs.get('len_starts', 0)
        self.len_ends = kwargs.get('len_ends', 0)
        self.register_arguments()
        self.register_op()

    def register_arguments(self):
         self.starts = [self.register_argument('starts[{}]'.format(i))
                for i in range(self.len_starts)]
         self.ends = [self.register_argument('ends[{}]'.format(i))
                for i in range(self.len_ends)]

    def register_op(self):
        self.op_meta = {
            'op_type': 'Crop',
            'n_inputs': 1, 'n_outputs': 1,
            'arguments': {
                'starts_desc': [e for e in self.starts] if len(self.starts) > 0 else None,
                'ends_desc': [e for e in self.ends] if len(self.ends) > 0 else None,
            }
        }

    def forward(self, x, starts, ends):
        inputs = [x]; self.unify_devices(inputs)
        outputs = [self.register_output(x.dtype)]
        for ix, d in enumerate(starts):
            self.set_argument_i(self.starts[ix], d)
        for ix, d in enumerate(ends):
            self.set_argument_i(self.ends[ix], d)
        return self.run(inputs, outputs)