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

from dragon.vm.torch.autograd import function


class ParamUpdate(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(ParamUpdate, self).__init__(key, dev, **kwargs)
        self.slot = kwargs.get('slot', '')
        self.lr_mult = kwargs.get('lr_mult', 1.)
        self.decay_mult = kwargs.get('decay_mult', 1.)
        self.op_type = kwargs.get('op_type', 'Update')

    def attributes(self):
        return {
            'op_type': self.op_type,
            'arguments': {
                'lr_mult': self.lr_mult,
                'decay_mult': self.decay_mult,
                'slot': self.slot,
            },
        }

    def forward(self, param, grad):
        self._check_device([param, grad])
        return self.dispatch(
            [grad], [param],
            no_grad=True,
            check_device=False,
        )


class GradAccumulate(function.Function):
    def __init__(self, key, dev, **kwargs):
        super(GradAccumulate, self).__init__(key, dev, **kwargs)

    def attributes(self):
        return {
            'op_type': 'Accumulate',
            'arguments': {
                'alpha': 1.,
                'beta': 1.,
            },
        }

    def forward(self, grads):
        outputs = [grad.id + '[acc]' for grad in grads]
        return self.dispatch(grads, outputs, no_grad=True)