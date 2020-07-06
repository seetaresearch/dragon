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

"""The Adam optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.training import optimizer


class Adam(optimizer.Optimizer):
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
        base_lr=0.001,
        beta1=0.9,
        beta2=0.999,
        eps=1e-8,
        **kwargs
    ):
        r"""Create an ``Adam`` updater.

        Parameters
        ----------
        base_lr : float, optional, default=0.001
            The initial value for :math:`\text{lr}`.
        beta1 : float, optional, default=0.9
            The initial value for :math:`\beta_{1}`.
        beta2 : float, optional, default=0.999
            The initial value for :math:`\beta_{2}`.
        eps : float, optional=1e-8
            The initial value for :math:`\epsilon`

        """
        super(Adam, self).__init__(**kwargs)
        self._init_set_defaults({
            'base_lr': base_lr,
            'beta1': beta1,
            'beta2': beta2,
            'eps': eps,
        })
