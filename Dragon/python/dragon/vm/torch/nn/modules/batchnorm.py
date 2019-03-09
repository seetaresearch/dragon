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

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.nn import Module, Parameter
from dragon.vm.torch.ops.creation import zeros, ones
from dragon.vm.torch.module import RunOperator


class _BatchNorm(Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.1,
                 affine=True, track_running_stats=True):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(Tensor(num_features))
            self.bias = Parameter(Tensor(num_features))
        else:
            self.register_buffer('weight', ones(num_features))
            self.register_buffer('bias', zeros(num_features))
        self.register_buffer('running_mean', zeros(num_features))
        self.register_buffer('running_var', ones(num_features))
        self.inputs = [self.running_mean, self.running_var, self.weight, self.bias]
        self.reset_parameters()
        self.register_op()
        self.op_metas = {'TRAIN': None, 'TEST': None}

    def reset_parameters(self):
        if self.affine:
            self.weight.data.uniform_()
            self.bias.data.zero_()

    def register_op(self):
        self.op_meta = {
            'op_type': 'BatchNorm',
            'arguments': {
                'axis': 1, # Data format: NCHW
                'momentum': 1. - self.momentum,
                'eps': self.eps,
                'use_stats': -1, # Meaningless
            }
        }

    def extra_repr(self):
        return '{num_features}, eps={eps}, momentum={momentum}, affine={affine}, ' \
               'track_running_stats={track_running_stats}'.format(**self.__dict__)

    def make_meta_from_phase(self, phase):
        """Make the custom meta by referring the phase and ctx.

        We extend this method as the original module can only
        detect the mutation of ctx(i.e. cpu -> cuda),
        but not the (train -> test).

        """
        def reset_meta(self, phase):
            self._module_key = None
            _ = self.module_key
            self._module_key += '/{}'.format(phase)
            self.op_meta['arguments']['use_stats'] = 0 \
                if phase == 'TRAIN' else 1
            self._gen_def()
            self.op_metas[phase] = (self._module_key, self._def)

        if self._module_key is None:
            # Init or Context has changed
            reset_meta(self, phase)
        else:
            # Context unchanged
            if self.op_metas[phase] is None:
                reset_meta(self, phase)

        return self.op_metas[phase]

    def forward(self, input):
        inputs = [input] + self.inputs
        self.unify_devices(inputs)
        outputs = [self.register_output()]
        phase = 'TRAIN' if self.training else 'TEST'
        # Normalize the input by using batch stats ALWAYS
        # Note that the update of moving average is meaningless(
        # Because we can not remove it. Why? Ask nvidia and cuDNN -:)
        if not self.track_running_stats: phase = 'TRAIN'
        meta = self.make_meta_from_phase(phase)
        return RunOperator(inputs, outputs, meta)


class BatchNorm1d(_BatchNorm):
    """Dragon does not use separate backend functions."""
    pass


class BatchNorm2d(_BatchNorm):
    """Dragon does not use separate backend functions."""
    pass


class BatchNorm3d(_BatchNorm):
    """Dragon does not use separate backend functions."""
    pass