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
"""Loss modules."""

from dragon.vm.torch.core.nn import functional
from dragon.vm.torch.core.nn.modules import utils
from dragon.vm.torch.core.nn.modules.module import Module


class _Loss(Module):
    """Base loss module."""

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super(_Loss, self).__init__()
        self.reduction = utils._get_loss_reduction(size_average, reduce, reduction)


class _WeightedLoss(_Loss):
    """Base weighted loss module."""

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
    ):
        super(_WeightedLoss, self).__init__(size_average, reduce, reduction)
        self.register_buffer("weight", weight)


class CTCLoss(_Loss):
    r"""Compute ctc loss with batched labels.
    `[Graves & Gomez, 2006] <http://www.cs.utoronto.ca/~graves/icml_2006.pdf>`_.

    Examples:

    ```python
    # t: num_steps
    # n: batch_size
    # c: num_classes(with blank at 0)
    t, n, c = 8, 4, 5
    m = torch.nn.CTCLoss(padding_mask=-1).cuda()
    logits = torch.ones(t, n, c)
    labels = torch.tensor([[1, 2, 3],
                           [1, 2, -1],
                           [1, -1, -1],
                           [1, 1, 1]], dtype='int32')
    loss = m(logits, labels)
    ```

    See Also
    --------
    `torch.nn.functional.ctc_loss(...)`_

    """

    def __init__(self, padding_mask=-1, reduction="mean"):
        """Create ``CTCLoss`` module.

        Parameters
        ----------
        padding_mask : int, optional, default=-1
            The mask for padding the redundant labels.
        reduction : {'none', 'mean', 'sum'}, optional
            The reduce method.

        """
        super(CTCLoss, self).__init__(reduction=reduction)
        self.padding_mask = padding_mask

    def forward(self, input, target):
        return functional.ctc_loss(
            input,
            target,
            padding_mask=self.padding_mask,
            reduction=self.reduction,
        )


class NLLLoss(_WeightedLoss):
    r"""Compute negative likelihood loss.

    The NLL loss function is defined as:

    .. math:: \text{NLLLoss}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    m1 = torch.nn.LogSoftmax(dim=1)
    m2 = torch.nn.NLLLoss()
    loss = m2(m1(torch.randn(2, 2)), torch.tensor([0, 1]))
    ```

    See Also
    --------
    `torch.nn.functional.nll_loss(...)`_

    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        ignore_index=None,
        reduce=None,
        reduction="mean",
    ):
        """Create a ``NLLLoss`` module.

        Parameters
        ----------
        weight : dragon.vm.torch.Tensor, optional
            The weight for each class.
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        ignore_index : int, optional
            The ignored value of target.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'mean', 'sum', 'valid'}, optional
            The reduce method.

        """
        super(NLLLoss, self).__init__(weight, size_average, reduce, reduction)
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return functional.nll_loss(
            input,
            target,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )


class BCEWithLogitsLoss(_WeightedLoss):
    r"""Compute sigmoid cross entropy.

    Examples:

    ```python
    m = torch.nn.BCEWithLogitsLoss()
    loss = m(torch.randn(2, 1), torch.tensor([0., 1.], 'float32'))
    ```

    See Also
    --------
    `torch.nn.functional.binary_cross_entropy_with_logits(...)`_

    """

    def __init__(
        self,
        weight=None,
        size_average=None,
        reduce=None,
        reduction="mean",
        pos_weight=None,
    ):
        """Create a ``BCEWithLogitsLoss`` module.

        Parameters
        ----------
        weight : dragon.vm.torch.Tensor, optional
            The weight for each class.
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'mean', 'sum', 'valid'}, optional
            The reduce method.

        """
        super(BCEWithLogitsLoss, self).__init__(weight, size_average, reduce, reduction)

    def forward(self, input, target):
        return functional.binary_cross_entropy_with_logits(input, target, reduction=self.reduction)


class CrossEntropyLoss(_WeightedLoss):
    r"""Compute softmax cross entropy.

    The **CrossEntropy** function is defined as:

    .. math:: \text{CrossEntropy}(p_{t}) = -\log(p_{t})

    Examples:

    ```python
    m = torch.nn.CrossEntropyLoss()
    logits = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    loss = m(logits, targets)
    ```

    See Also
    --------
    `torch.nn.functional.cross_entropy(...)`_

    """

    def __init__(
        self,
        dim=1,
        weight=None,
        size_average=None,
        ignore_index=None,
        reduce=None,
        reduction="mean",
    ):
        """Create a ``CrossEntropyLoss`` module.

        Parameters
        ----------
        dim : int, optional, default=1
            The dimension to reduce.
        weight : dragon.vm.torch.Tensor, optional
            The weight for each class.
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        ignore_index : int, optional
            The ignored value of target.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'mean', 'sum', 'valid'}, optional
            The reduce method.

        """
        super(CrossEntropyLoss, self).__init__(weight, size_average, reduce, reduction)
        self.dim = dim
        self.ignore_index = ignore_index

    def forward(self, input, target):
        return functional.cross_entropy(
            input,
            target,
            dim=self.dim,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )


class KLDivLoss(_Loss):
    """Compute Kullback-Leibler divergence.

    Examples:

    ```python
    m = torch.nn.KLDivLoss()
    eps = 1e-12  # Epsilon to avoid log(0)
    # Compute KL(P || Q)
    q = torch.tensor([0.0, 0.1, 0.2, 0.3, 1.0])
    p = torch.tensor([0.0, 0.3, 0.2, 0.1, 0.9])
    loss = m(torch.log(torch.clamp(q, eps)), torch.clamp(p, eps))
    ```

    See Also
    --------
    `torch.nn.functional.kl_div(...)`_

    """

    def __init__(
        self,
        size_average=None,
        reduce=None,
        reduction="mean",
        log_target=False,
    ):
        """Create a ``KDivLoss`` module.

        Parameters
        ----------
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'batchmean', 'mean', 'sum'}, optional
            The reduce method.
        log_target : bool, optional, default=False
            The flag indicating whether ``target`` is passed in log space.

        """
        super(KLDivLoss, self).__init__(size_average, reduce, reduction)
        self.log_target = log_target

    def forward(self, input, target):
        return functional.kl_div(
            input,
            target,
            reduction=self.reduction,
            log_target=self.log_target,
        )


class L1Loss(_Loss):
    r"""Compute element-wise absolute value difference.

    The ``L1Loss`` function is defined as:

    .. math:: \text{L1Loss}(x, y) = |x - y|

    Examples:

    ```python
    m = torch.nn.L1Loss()
    loss = m(torch.ones(2, 3), torch.zeros(2, 3))
    ```

    See Also
    --------
    `torch.nn.functional.l1_loss(...)`_

    """

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        """Create a ``L1Loss`` module.

        Parameters
        ----------
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'mean', 'sum'}, optional
            The reduce method.

        """
        super(L1Loss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return functional.l1_loss(input, target, reduction=self.reduction)


class MSELoss(_Loss):
    r"""Compute element-wise squared error.

    The ``MSELoss`` function is defined as:

    .. math:: \text{MSELoss}(x, y) = (x - y)^{2}

    Examples:

    ```python
    m = torch.nn.MSELoss()
    loss = m(torch.ones(2, 3) * 2, torch.zeros(2, 3))
    ```

    See Also
    --------
    `torch.nn.functional.mse_loss(...)`_

    """

    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        """Create a ``MSELoss`` module.

        Parameters
        ----------
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'mean', 'sum'}, optional
            The reduce method.

        """
        super(MSELoss, self).__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return functional.mse_loss(input, target, reduction=self.reduction)


class SmoothL1Loss(_Loss):
    r"""Compute element-wise error transited from L1 and L2.
    `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The **SmoothL1Loss** function is defined as:

    .. math::
        \text{SmoothL1Loss}(x, y) =
            \begin{cases}
                0.5 * (x - y)^{2} / beta, & \text{ if } |x - y| < beta \\
                |x - y| - 0.5 * beta, & \text{ otherwise }
            \end{cases}

    Examples:

    ```python
    m = torch.nn.SmoothL1Loss(beta=0.11)
    loss = m(torch.ones(2, 3), torch.zeros(2, 3))
    ```

    See Also
    --------
    `torch.nn.functional.smooth_l1_loss(...)`_

    """

    def __init__(
        self,
        beta=1.0,
        size_average=None,
        reduce=None,
        reduction="mean",
    ):
        """Create a ``SmoothL1Loss`` module.

        Parameters
        ----------
        beta : float, optional, default=1.
            The transition point from L1 to L2 loss
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'mean', 'sum'}, optional
            The reduce method.

        """
        super(SmoothL1Loss, self).__init__(size_average, reduce, reduction)
        self.beta = beta

    def forward(self, input, target):
        return functional.smooth_l1_loss(
            input,
            target,
            beta=self.beta,
            reduction=self.reduction,
        )


class SigmoidFocalLoss(_WeightedLoss):
    r"""Compute sigmoid focal loss.
    `[Lin et.al, 2017] <https://arxiv.org/abs/1708.02002>`__.

    The **FocalLoss** function is defined as:

    .. math:: \text{FocalLoss}(p_{t}) = -(1 - p_{t})^{\gamma}\log(p_{t})

    Examples:

    ```python
    m = torch.nn.SigmoidFocalLoss()
    logits = torch.randn(2, 2)
    targets = torch.tensor([0, 1])
    loss = m(logits, targets)
    ```

    See Also
    --------
    `torch.nn.functional.sigmoid_focal_loss(...)`_

    """

    def __init__(
        self,
        alpha=0.25,
        gamma=2.0,
        weight=None,
        size_average=None,
        start_index=0,
        reduce=None,
        reduction="mean",
    ):
        """Create a ``SigmoidFocalLoss`` module.

        Parameters
        ----------
        alpha : float, optional, default=0.25
            The scale factor on the rare class.
        gamma : float, optional, default=2.
            The exponential decay factor on the easy examples.
        weight : dragon.vm.torch.Tensor, optional
            The weight for each class.
        size_average : bool, optional
            ``True`` to set the ``reduction`` to *'mean'*.
        start_index : int, optional, default=0
            The start value of target.
        reduce : bool, optional
            ``True`` to set the ``reduction`` to *'sum'* or *'mean'*.
        reduction : {'none', 'mean', 'sum', 'valid'}, optional
            The reduce method.

        """
        super(SigmoidFocalLoss, self).__init__(weight, size_average, reduce, reduction)
        self.alpha, self.gamma = alpha, gamma
        self.start_index = start_index

    def forward(self, input, target):
        return functional.sigmoid_focal_loss(
            input,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            reduction=self.reduction,
            start_index=self.start_index,
        )
