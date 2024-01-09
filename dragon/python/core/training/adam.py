# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Adam optimizers."""

from dragon.core.training import optimizer


class Adam(optimizer.Optimizer):
    r"""The optimizer to apply Adam algorithm.
    `[Kingma & Ba, 2014] <https://arxiv.org/abs/1412.6980>`_.

    The **Adam** update is defined as:

    .. math::
        \text{Adam}(g) = \text{lr} * (\frac{\text{correction}* m_{t}}
                                           {\sqrt{v_{t}} + \epsilon}) \\
            \quad \\ \text{where}\quad
                \begin{cases}
                    \text{correction} = \sqrt{1 - \beta_{2}^{t}} / (1 - \beta_{1}^{t}) \\
                    m_{t} = \beta_{1} * m_{t-1} + (1 - \beta_{1}) * g \\
                    v_{t} = \beta_{2} * v_{t-1} + (1 - \beta_{2}) * g^{2}
                \end{cases}

    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, **kwargs):
        r"""Create an ``Adam`` updater.

        Parameters
        ----------
        lr : float, optional, default=0.001
            The initial value to :math:`\text{lr}`.
        beta1 : float, optional, default=0.9
            The initial value to :math:`\beta_{1}`.
        beta2 : float, optional, default=0.999
            The initial value to :math:`\beta_{2}`.
        eps : float, optional=1e-8
            The initial value to :math:`\epsilon`

        """
        super(Adam, self).__init__(**kwargs)
        self._set_hyper("lr", lr)
        self._set_hyper("beta1", beta1)
        self._set_hyper("beta2", beta2)
        self._set_hyper("eps", eps)


class AdamW(Adam):
    r"""The optimizer to apply AdamW algorithm.
    `[Loshchilov & Hutter, 2017] <https://arxiv.org/abs/1711.05101>`_.

    The **AdamW** update is defined as:

    .. math::
        \text{AdamW}(g, p) = \text{lr} * (\frac{\text{correction} * m_{t}}
                                               {\sqrt{v_{t}} + \epsilon} + \lambda p) \\
            \quad \\ \text{where}\quad
                \begin{cases}
                    \text{correction} = \sqrt{1 - \beta_{2}^{t}} / (1 - \beta_{1}^{t}) \\
                    m_{t} = \beta_{1} * m_{t-1} + (1 - \beta_{1}) * g \\
                    v_{t} = \beta_{2} * v_{t-1} + (1 - \beta_{2}) * g^{2} \\
                \end{cases}

    """

    def __init__(self, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01, **kwargs):
        r"""Create an ``AdamW`` updater.

        Parameters
        ----------
        lr : float, optional, default=0.001
            The initial value to :math:`\text{lr}`.
        beta1 : float, optional, default=0.9
            The initial value to :math:`\beta_{1}`.
        beta2 : float, optional, default=0.999
            The initial value to :math:`\beta_{2}`.
        eps : float, optional, default=1e-8
            The initial value to :math:`\epsilon`
        weight_decay : float, optional, default=0.01
            The initial value to :math:`\lambda`.

        """
        super(AdamW, self).__init__(lr, beta1, beta2, eps, weight_decay=weight_decay, **kwargs)
