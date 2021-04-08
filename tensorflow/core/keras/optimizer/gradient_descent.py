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
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/optimizer_v2/gradient_descent.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.tensorflow.core.keras.optimizer import optimizer


class SGD(optimizer.Optimizer):
    r"""The optimizer to apply SGD algorithm.

    Following SGD algorithms are supported:

    **VanillaSGD**, whose update is defined as:

    .. math:: \text{VanillaSGD}(g) = -\text{lr} * g

    **MomentumSGD**
    `[Polyak, 1964] <https://doi.org/10.1016/0041-5553(64)90137-5>`_,
    whose update is defined as:

    .. math:: \text{MomentumSGD}(g) =
            -(\text{momentum} * m_{t-1} + \text{lr} * g)

    **NesterovSGD**
    `[Sutskever et.al, 2013] <http://www.cs.toronto.edu/~hinton/absps/momentum.pdf>`_,
    whose update is defined as:

    .. math:: \text{NesterovSGD}(g) =
            -((1 + \text{momentum}) * m_{t} - \text{momentum} * m_{t-1}) \\
        \quad \\ \text{where} \quad
            m_{t} = \text{momentum} * m_{t-1} + \text{lr} * g

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

    def __init__(
        self,
        learning_rate=0.01,
        momentum=0.0,
        nesterov=False,
        name=None,
        **kwargs
    ):
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
        self._set_hyper('lr', learning_rate)
        self._set_hyper('momentum', momentum)
        self._op_type = ('Nesterov' if nesterov else 'SGD') + 'Update'
        self._hyper_aliases['learning_rate'] = 'lr'
