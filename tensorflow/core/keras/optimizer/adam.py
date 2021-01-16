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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/adam.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.tensorflow.core.keras.optimizer import optimizer


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
        learning_rate=0.001,
        beta_1=0.9,
        beta_2=0.999,
        epsilon=1e-7,
        name=None,
        **kwargs
    ):
        r"""Create an ``Adam`` optimizer.

        Parameters
        ----------
        learning_rate : float, optional, default=0.001
            The initial value to :math:`\text{lr}`.
        beta_1 : float, optional, default=0.9
            The initial value to :math:`\beta_{1}`.
        beta_2 : float, optional, default=0.999
            The initial value to :math:`\beta_{2}`.
        epsilon : float, optional, default=1e-7
            The initial value to :math:`\epsilon`.
        name : str, optional
            The optional optimizer name.

        """
        super(Adam, self).__init__(name, **kwargs)
        self._set_hyper('learning_rate', kwargs.get('lr', learning_rate), 'base_lr')
        self._set_hyper('beta_1', beta_1, 'beta1')
        self._set_hyper('beta_2', beta_2, 'beta2')
        self._set_hyper('epsilon', epsilon, 'eps')
