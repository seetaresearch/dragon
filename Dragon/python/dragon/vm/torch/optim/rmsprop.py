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
#      <https://github.com/pytorch/pytorch/blob/master/torch/optim/rmsprop.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    def __init__(self, params, lr=1e-2, alpha=0.99, eps=1e-8, weight_decay=0,
                 momentum=0, centered=False, scale_gradient=1.0, clip_gradient=-1.0):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if momentum != 0:
            raise NotImplementedError()
        if not 0.0 <= alpha:
            raise ValueError("Invalid alpha value: {}".format(alpha))

        defaults = dict(lr=lr, momentum=momentum, alpha=alpha, eps=eps,
                        centered=centered, weight_decay=weight_decay,
                        scale_gradient=scale_gradient, clip_gradient=clip_gradient)
        super(RMSprop, self).__init__(params, defaults)
        self._update_type = 'RMSPropUpdate'
        self._mutable_parameters = {
            'lr': 'base_lr',
            'alpha': 'decay',
            'eps': 'eps',
            'weight_decay': 'l2_decay',
            'clip_gradient': 'clip_gradient',
            'scale_gradient': 'scale_gradient',
        }