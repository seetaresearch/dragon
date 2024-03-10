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

from dragon.vm.torch.core.optim.optimizer import Optimizer
from dragon.vm.torch.core.optim.zero import ZeroOptimizer


class Adam(Optimizer):
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

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0,
        amsgrad=False,
        **kwargs,
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
            ``True`` to switch to **AMSGrad** optimizer.

        """
        if not 0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= eps:
            raise ValueError("Invalid epsilon: {}".format(eps))
        if not 0 <= betas[0] < 1:
            raise ValueError("Invalid beta1: {}".format(betas[0]))
        if not 0 <= betas[1] < 1:
            raise ValueError("Invalid beta2: {}".format(betas[1]))
        if amsgrad:
            raise NotImplementedError
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            amsgrad=amsgrad,
            weight_decay=weight_decay,
        )
        super(Adam, self).__init__(params, defaults, **kwargs)
        self._hyper_dict.pop("amsgrad")  # Unsupported.


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

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        **kwargs,
    ):
        r"""Create an ``AdamW`` optimizer.

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
        weight_decay : float, optional, default=0.01
            The initial value to :math:`\lambda`.
        amsgrad : bool, optional, default=False
            ``True`` to switch to **AMSGrad** optimizer.

        """
        super(AdamW, self).__init__(
            params,
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay,
            amsgrad=amsgrad,
            **kwargs,
        )


class ZeroAdamW(ZeroOptimizer):
    r"""The optimizer to apply AdamW algorithm with ZeRO.
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

    def __init__(
        self,
        params,
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        amsgrad=False,
        **kwargs,
    ):
        r"""Create an ``ZeroAdamW`` optimizer.

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
        weight_decay : float, optional, default=0.01
            The initial value to :math:`\lambda`.
        amsgrad : bool, optional, default=False
            ``True`` to switch to **AMSGrad** optimizer.

        """
        defaults = dict(
            lr=lr,
            beta1=betas[0],
            beta2=betas[1],
            eps=eps,
            weight_decay=weight_decay,
        )
        super(ZeroAdamW, self).__init__(params, defaults, **kwargs)
