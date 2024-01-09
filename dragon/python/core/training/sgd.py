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
"""SGD optimizers."""

from dragon.core.training import optimizer


class SGD(optimizer.Optimizer):
    r"""The optimizer to apply SGD algorithm.

    Following SGD algorithms are supported:

    **VanillaSGD**, whose update is defined as:

    .. math:: \text{VanillaSGD}(g) = \text{lr} * g

    **MomentumSGD**
    `[Polyak, 1964] <https://doi.org/10.1016/0041-5553(64)90137-5>`_,
    whose update is defined as:

    .. math:: \text{MomentumSGD}(g) = \text{lr} * m_{t} \\
        \quad \\ \text{where} \quad m_{t} = \text{momentum} * m_{t-1} + g

    **NesterovSGD**
    `[Sutskever et.al, 2013] <http://www.cs.toronto.edu/~hinton/absps/momentum.pdf>`_,
    whose update is defined as:

    .. math:: \text{NesterovSGD}(g) = \text{lr} * (\text{momentum} * m_{t} + g) \\
        \quad \\ \text{where} \quad m_{t} = \text{momentum} * m_{t-1} + g

    """

    def __init__(self, lr=0.01, momentum=0.9, nesterov=False, **kwargs):
        r"""Create a ``SGD`` updater.

        Parameters
        ----------
        lr : float, optional, default=0.01
            The initial value to :math:`\text{lr}`.
        momentum : float, optional, default=0.9
            The initial value to :math:`\text{momentum}`.
        nesterov : bool, optional, default=False
            ``True`` to switch to **NesterovSGD** optimizer.

        """
        super(SGD, self).__init__(**kwargs)
        self._op_type = "NesterovSGD" if nesterov else "MomentumSGD"
        self._set_hyper("lr", lr)
        self._set_hyper("momentum", momentum)
