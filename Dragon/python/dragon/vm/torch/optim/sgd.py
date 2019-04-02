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
#      <https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.optim.optimizer import Optimizer, required


class SGD(Optimizer):
    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=-1.,
        nesterov=False,
        scale_gradient=1.,
        clip_gradient=-1.,
    ):
        if lr is not required and lr < 0.:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            scale_gradient=scale_gradient,
            clip_gradient=clip_gradient,
        )
        if nesterov and (momentum <= 0. or dampening != 0.):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening.")
        super(SGD, self).__init__(params, defaults)
        self._update_type = 'NesterovUpdate' if nesterov else 'SGDUpdate'
        self._mutable_parameters = {
            'lr': 'base_lr',
            'momentum': 'momentum',
            'weight_decay': 'l2_decay',
            'clip_gradient': 'clip_gradient',
            'scale_gradient': 'scale_gradient',
        }