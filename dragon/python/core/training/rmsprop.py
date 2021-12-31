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
"""RMSprop optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.training import optimizer


class RMSprop(optimizer.Optimizer):
    r"""The optimizer to apply RMSprop algorithm.
    `[Hinton et.al, 2013] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.

    The **RMSprop** update is defined as:

    .. math::
        \text{RMSprop}(g) = \text{lr} * m_{t} \\
            \quad \\ \text{where} \quad
                \begin{cases}
                    v_{t} = \alpha * v_{t-1} + (1 - \alpha) * g^{2} \\
                    m_{t} = \text{momentum} * m_{t-1} + \frac{g}{\sqrt{v_{t}} + \epsilon}
                \end{cases}

    """

    def __init__(self, lr=0.01, momentum=0, alpha=0.9, eps=1e-8, **kwargs):
        r"""Create a ``RMSProp`` optimizer.

        Parameters
        ----------
        lr : float, optional, default=0.01
            The initial value to :math:`\text{lr}`.
        momentum : float, optional, default=0
            The initial value to :math:`\text{momentum}`.
        alpha : float, optional, default=0.9
            The initial value to :math:`\alpha`.
        eps : float, optional, default=1e-8
            The initial value to :math:`\epsilon`.

        """
        super(RMSprop, self).__init__(**kwargs)
        self._set_hyper('lr', lr)
        self._set_hyper('momentum', momentum)
        self._set_hyper('alpha', alpha)
        self._set_hyper('eps', eps)
