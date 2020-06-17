# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""The optimizer which implements Adam algorithm.
    `[Kingma & Ba, 2014] <https://arxiv.org/abs/1412.6980>`_.

    The **Adam** update is defined as:

    .. math::
        \text{Adam}(g) = -\frac{\text{lr} * m_{t}}{\sqrt{v_{t}} + \epsilon} \\
            \quad \\ \text{where}\quad
                \begin{cases}
                    m_{t} = \beta_{1} * m_{t-1} + (1 - \beta_{1}) * g \\
                    v_{t} = \beta_{2} * v_{t-1} + (1 - \beta_{2}) * g^{2}
                \end{cases}

    """

    def __init__(
        self,
        params,
        lr=1e-3,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        scale_gradient=1.,
        clip_gradient=-1.,
    ):
        r"""Create an ``Adam`` optimizer.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        lr : float, required
            The initial value for :math:`\text{lr}`.
        beta1 : float, optional, default=0.9
            The initial value for :math:`\beta_{1}`.
        beta2 : float, optional, default=0.999
            The initial value for :math:`\beta_{2}`.
        eps : float, optional, default=1e-8
            The initial value of :math:`\epsilon`.
        weight_decay : float, optional, default=-1.
            The factor of L2 penalty.
        amsgrad : bool, optional, default=False
            **True** to switch to **AMSGrad** optimizer.
        scale_gradient : float, optional, default=1.
            The factor to scale gradients.
        clip_gradient : float, optional, default=-1.
            The norm thresh to clip gradients.

        """
        if not 0. <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0. <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0. <= beta1 < 1.:
            raise ValueError("Invalid beta parameter at index 0: {}".format(beta1))
        if not 0. <= beta2 < 1.:
            raise ValueError("Invalid beta parameter at index 1: {}".format(beta2))
        if amsgrad:
            raise NotImplementedError()
        defaults = dict(
            lr=lr,
            beta1=beta1,
            beta2=beta2,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            scale_gradient=scale_gradient,
            clip_gradient=clip_gradient,
        )
        super(Adam, self).__init__(params, defaults)
        self._update_op_type = 'AdamUpdate'
        self._shared_args = {
            'lr': 'base_lr',
            'beta1': 'beta1',
            'beta2': 'beta2',
            'eps': 'eps',
            'weight_decay': 'l2_decay',
            'clip_gradient': 'clip_gradient',
            'scale_gradient': 'scale_gradient',
        }
