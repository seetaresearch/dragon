# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    def __init__(self, params, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8,
                 weight_decay=0, amsgrad=False, scale_gradient=1.0, clip_gradient=-1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= beta1 < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0.0 <= beta2 < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))
        if amsgrad:
            raise NotImplementedError()
        defaults = dict(lr=lr, beta1=beta1, beta2=beta2, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad,
                        scale_gradient=scale_gradient, clip_gradient=clip_gradient)
        super(Adam, self).__init__(params, defaults)
        self._update_type = 'AdamUpdate'
        self._mutable_parameters = {
            'lr': 'base_lr',
            'beta1': 'beta1',
            'beta2': 'beta2',
            'eps': 'eps',
            'weight_decay': 'l2_decay',
            'clip_gradient': 'clip_gradient',
            'scale_gradient': 'scale_gradient',
        }