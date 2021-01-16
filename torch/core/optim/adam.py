# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/pytorch/pytorch/blob/master/torch/optim/adam.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.optim.optimizer import Optimizer


class Adam(Optimizer):
    r"""The optimizer to apply Adam algorithm.
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
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        scale=1,
        clip_norm=0,
    ):
        r"""Create an ``Adam`` optimizer.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        lr : float, required
            The initial value to :math:`\text{lr}`.
        betas : Tuple[float, float], optional, default=(0.9, 0.999)
            The initial value to :math:`\beta_{1}` and :math:`\beta_{2}`.
        eps : float, optional, default=1e-8
            The initial value to :math:`\epsilon`.
        weight_decay : float, optional, default=0
            The L2 penalty factor to weight.
        amsgrad : bool, optional, default=False
            **True** to switch to **AMSGrad** optimizer.
        scale : float, optional, default=1
            The scaling factor to gradient.
        clip_norm : float, optional, default=0
            The maximum L2 norm to clip gradient.

        """
        if not 0. <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0. <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0. <= betas[0] < 1.:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0. <= betas[1] < 1.:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if amsgrad:
            raise NotImplementedError
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            amsgrad=amsgrad,
            scale=scale,
            clip_norm=clip_norm,
            weight_decay=weight_decay,
        )
        super(Adam, self).__init__(params, defaults)
        self._shared_args = {
            'lr': 'base_lr',
            'beta1': 'beta1',
            'beta2': 'beta2',
            'eps': 'eps',
            'scale': 'scale',
            'clip_norm': 'clip_norm',
            'weight_decay': 'weight_decay',
        }
