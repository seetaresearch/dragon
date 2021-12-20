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
#     <https://github.com/pytorch/pytorch/blob/master/torch/optim/sgd.py>
#
# ------------------------------------------------------------
"""SGD optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.vm.torch.core.optim.optimizer import Optimizer
from dragon.vm.torch.core.optim.optimizer import required


class SGD(Optimizer):
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

    You can use one of them by setting the defaults:

    ```python
    # Set the ``lr`` only
    vanilla_sgd = torch.optim.SGD(lr=0.1)

    # Set the ``lr`` and ``momentum``
    momentum_sgd = torch.optim.SGD(lr=0.1, momentum=0.9)

    # Set the ``lr``, ``momentum`` and ``nesterov``
    nesterov_sgd = torch.optim.SGD(lr=0.1, momentum=0.9, nesterov=True)
    ```

    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        dampening=0,
        weight_decay=0,
        nesterov=False,
        **kwargs
    ):
        r"""Create a ``SGD`` optimizer.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        lr : float, required
            The initial value to :math:`\text{lr}`.
        momentum : float, optional, default=0
            The initial value to :math:`\text{momentum}`.
        dampening : float, optional, default=0
            The dampening for :math:`\text{momentum}`.
        weight_decay : float, optional, default=0
            The L2 penalty factor to weight.
        nesterov : bool, optional, default=False
            ``True`` to switch to **NesterovSGD** optimizer.

        """
        if lr is not required and lr < 0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0:
            raise ValueError('Invalid momentum: {}'.format(momentum))
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError('Nesterov momentum requires a momentum and zero dampening.')
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening,
                        nesterov=nesterov, weight_decay=weight_decay)
        super(SGD, self).__init__(params, defaults, **kwargs)
        self._op_type = 'NesterovSGD' if nesterov else 'MomentumSGD'
        self._hyper.pop('dampening')  # Unsupported.


class LARS(Optimizer):
    r"""The optimizer to apply LARS algorithm.
    `[You et.al, 2017] <https://arxiv.org/abs/1708.03888>`_.

    The **LARS** update is defined as:

    .. math:: \text{LARS}(g, p) = \text{lr} * m_{t} \\
        \quad \\ \text{where} \quad m_{t} =
            \text{momentum} * m_{t-1} +
            \eta * \frac{\lVert p \rVert}{\lVert g \rVert}

    """

    def __init__(
        self,
        params,
        lr=required,
        momentum=0,
        trust_coef=0.001,
        weight_decay=0,
        **kwargs
    ):
        r"""Create a ``LARS`` optimizer.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        lr : float, required
            The initial value to :math:`\text{lr}`.
        momentum : float, optional, default=0
            The initial value to :math:`\text{momentum}`.
        trust_coef : float, optional, default=0.001
            The initial value to :math:`\eta`.
        weight_decay : float, optional, default=0
            The L2 penalty factor to weight.

        """
        if lr is not required and lr < 0:
            raise ValueError('Invalid learning rate: {}'.format(lr))
        if momentum < 0:
            raise ValueError('Invalid momentum: {}'.format(momentum))
        if trust_coef < 0:
            raise ValueError('Invalid trust coefficient: {}'.format(trust_coef))
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay,
                        trust_coef=trust_coef)
        super(LARS, self).__init__(params, defaults, **kwargs)
