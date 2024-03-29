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
"""Gradient descent optimizers."""

from dragon.vm.keras.core.optimizer import optimizer


class SGD(optimizer.Optimizer):
    r"""The optimizer to apply SGD algorithm.

    Following SGD algorithms are supported:

    **VanillaSGD**, whose update is defined as:

    .. math:: \text{VanillaSGD}(g) = -\text{lr} * g

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

    You can use one of them by setting the defaults:

    ```python
    # Set the ``lr`` only
    vanilla_sgd = tf.keras.optimizers.SGD(learning_rate=0.1)

    # Set the ``lr`` and ``momentum``
    momentum_sgd = tf.keras.optimizers.SGD(learning_rate=0.1, momentum=0.9)

    # Set the ``lr``, ``momentum`` and ``nesterov``
    nesterov_sgd = tf.keras.optimizers.SGD(lr=0.1, momentum=0.9, nesterov=True)
    ```

    """

    def __init__(self, learning_rate=0.01, momentum=0.0, nesterov=False, name=None, **kwargs):
        r"""Create a ``SGD`` optimizer.

        Parameters
        ----------
        learning_rate : float, optional, default=0.01
            The initial value to :math:`\text{lr}`.
        momentum : float, optional, default=0
            The initial value to :math:`\text{momentum}`.
        nesterov : bool, optional, default=False
            ``True`` to switch to **NesterovSGD** optimizer.
        name : str, optional
            The optional optimizer name.

        """
        super(SGD, self).__init__(name, **kwargs)
        self._set_hyper("lr", learning_rate)
        self._set_hyper("momentum", momentum)
        self._op_type = "NesterovSGD" if nesterov else "MomentumSGD"
        self._hyper_aliases["learning_rate"] = "lr"
