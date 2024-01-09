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

from dragon.vm.keras.core.optimizer import optimizer


class RMSprop(optimizer.Optimizer):
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
        self, learning_rate=0.001, rho=0.9, momentum=0.0, epsilon=1e-7, name=None, **kwargs
    ):
        r"""Create a ``RMSprop`` optimizer.

        Parameters
        ----------
        learning_rate : float, optional, default=0.001
            The initial value to :math:`\text{lr}`.
        rho : float, optional, default=0.9
            The initial value to :math:`\alpha`.
        momentum : float, optional, default=0
            The initial value to :math:`\text{momentum}`.
        epsilon : float, optional, default=1e-7
            The initial value to :math:`\epsilon`.
        name : str, optional
            The optional optimizer name.

        """
        super(RMSprop, self).__init__(name, **kwargs)
        self._set_hyper("lr", learning_rate)
        self._set_hyper("momentum", momentum)
        self._set_hyper("alpha", rho)
        self._set_hyper("eps", epsilon)
        self._hyper_aliases["learning_rate"] = "lr"
        self._hyper_aliases["rho"] = "alpha"
        self._hyper_aliases["epsilon"] = "eps"
