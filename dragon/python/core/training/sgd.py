# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The SGD optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.training import optimizer


class SGD(optimizer.Optimizer):
    r"""The optimizer to apply MomentumSGD algorithm.
    `[Polyak, 1964] <https://doi.org/10.1016/0041-5553(64)90137-5>`_.

    The **MomentumSGD** update is defined as:

    .. math:: \text{MomentumSGD}(g) = -(\text{momentum} * m_{t-1} + \text{lr} * g)

    """

    def __init__(self, base_lr=0.01, momentum=0.9, **kwargs):
        r"""Create a ``SGD`` updater.

        Parameters
        ----------
        base_lr : float, optional, default=0.01
            The initial value for :math:`\text{lr}`.
        momentum : float, optional, default=0.9
            The initial value for :math:`\text{momentum}`.

        """
        super(SGD, self).__init__(**kwargs)
        self._init_set_defaults({
            'base_lr': base_lr,
            'momentum': momentum,
        })


class Nesterov(optimizer.Optimizer):
    r"""The optimizer to apply NesterovSGD algorithm.
    `[Sutskever et.al, 2013] <http://www.cs.toronto.edu/~hinton/absps/momentum.pdf>`_.

    The **NesterovSGD** update is defined as:

    .. math:: \text{NesterovSGD}(g) =
            -((1 + \text{momentum}) * m_{t} - \text{momentum} * m_{t-1}) \\
        \quad \\ \text{where} \quad
            m_{t} = \text{momentum} * m_{t-1} + \text{lr} * g

    """

    def __init__(self, base_lr=0.01, momentum=0.9, **kwargs):
        r"""Create a ``Nesterov`` optimizer.

        Parameters
        ----------
        base_lr : float, optional, default=0.01
            The initial value for :math:`\text{lr}`.
        momentum : float, optional, default=0.9
            The initial value for :math:`\text{momentum}`.

        """
        super(Nesterov, self).__init__(**kwargs)
        self._init_set_defaults({
            'base_lr': base_lr,
            'momentum': momentum,
        })
