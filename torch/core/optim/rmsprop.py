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
"""RMSprop optimizers."""

from dragon.vm.torch.core.optim.optimizer import Optimizer


class RMSprop(Optimizer):
    r"""The optimizer to apply RMSprop algorithm.
    `[Hinton et.al, 2013] <http://www.cs.utoronto.ca/~bonner/courses/2016s/csc321/lectures/lec6.pdf>`_.

    The **RMSprop** update is defined as:

    .. math::
        \text{RMSprop}(g) = \text{lr} * m_{t} \\
            \quad \\ \text{where}\quad
                \begin{cases}
                    v_{t} = \alpha * v_{t-1} + (1 - \alpha) * g^{2} \\
                    m_{t} = \text{momentum} * m_{t-1} +
                            \frac{g}{\sqrt{v_{t}} + \epsilon}
                \end{cases}

    """  # noqa: E501

    def __init__(
        self,
        params,
        lr=1e-2,
        alpha=0.99,
        eps=1e-8,
        weight_decay=0,
        momentum=0,
        centered=False,
        **kwargs
    ):
        r"""Create a ``RMSprop`` optimizer.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        lr : float, required
            The initial value to :math:`\text{lr}`.
        alpha : float, optional, default=0.99
            The initial value to :math:`\alpha`.
        eps : float, optional, default=1e-7
            The initial value to :math:`\epsilon`.
        weight_decay : float, optional, default=0
            The L2 penalty factor to weight.
        momentum : float, optional, default=0
            The initial value to :math:`\text{momentum}`.

        """
        if not 0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0 <= eps:
            raise ValueError("Invalid epsilon: {}".format(eps))
        if momentum < 0.0:
            raise ValueError("Invalid momentum: {}".format(momentum))
        if not 0 <= alpha:
            raise ValueError("Invalid alpha: {}".format(alpha))
        defaults = dict(
            lr=lr,
            momentum=momentum,
            alpha=alpha,
            eps=eps,
            centered=centered,
            weight_decay=weight_decay,
        )
        super(RMSprop, self).__init__(params, defaults, **kwargs)
        self._hyper_dict.pop("centered")  # Unsupported.
