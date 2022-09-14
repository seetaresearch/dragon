# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""BatchNorm modules."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import inspect

from dragon.core import distributed
from dragon.vm.torch.core.nn import functional as F
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.parameter import Parameter
from dragon.vm.torch.core.ops import constant_ops
from dragon.vm.torch.core.tensor import Tensor


class _BatchNorm(Module):
    """BatchNorm base module."""

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        super(_BatchNorm, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.affine = affine
        self.track_running_stats = track_running_stats
        if self.affine:
            self.weight = Parameter(Tensor(num_features))
            self.bias = Parameter(Tensor(num_features))
        else:
            self.register_buffer('weight', constant_ops.ones(num_features))
            self.register_buffer('bias', constant_ops.zeros(num_features))
        if self.track_running_stats:
            self.num_batches_tracked = 0
        else:
            self.num_batches_tracked = None
        self.register_buffer('running_mean', constant_ops.zeros(num_features))
        self.register_buffer('running_var', constant_ops.ones(num_features))
        self.reset_parameters()

    def reset_running_stats(self):
        if self.track_running_stats:
            self.running_mean.zero_()
            self.running_var.fill_(1)
            self.num_batches_tracked = 0

    def reset_parameters(self):
        self.reset_running_stats()
        if self.affine:
            self.weight.data.one_()
            self.bias.data.zero_()

    def extra_repr(self):
        return '{num_features}, ' \
               'eps={eps}, ' \
               'momentum={momentum}, ' \
               'affine={affine}, ' \
               'track_running_stats={track_running_stats}' \
               .format(**self.__dict__)

    def forward(self, input):
        return F.batch_norm(
            input,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=self.training,
            momentum=self._get_momentum(),
            eps=self.eps,
        )

    def _apply(self, fn):
        lambda_source = inspect.getsource(fn)
        if 'half_()' in lambda_source:
            return self  # Float32 parameters are required.
        return super(_BatchNorm, self)._apply(fn)

    def _get_momentum(self):
        """Return the current momentum value."""
        momentum = 0.0 if self.momentum is None else self.momentum
        if self.track_running_stats:
            if self.training:
                if self.num_batches_tracked is not None:
                    self.num_batches_tracked += 1
                if self.momentum is None:
                    momentum = 1.0 / float(self.num_batches_tracked)
        else:
            momentum = 0.0
        return momentum


class BatchNorm1d(_BatchNorm):
    r"""Apply the batch normalization over 2d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = (1 - \text{momentum}) * x_{\text{running}}
                                   + \text{momentum} * x_{\text{batch}}

    See Also
    --------
    `torch.nn.functional.batch_norm(...)`_

    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        r"""Create a ``BatchNorm1d`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.

        """
        super(BatchNorm1d, self).__init__(
            num_features,
            eps, momentum,
            affine, track_running_stats,
        )


class BatchNorm2d(_BatchNorm):
    r"""Apply the batch normalization over 3d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = (1 - \text{momentum}) * x_{\text{running}}
                                   + \text{momentum} * x_{\text{batch}}

    See Also
    --------
    `torch.nn.functional.batch_norm(...)`_

    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        r"""Create a ``BatchNorm2d`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.

        """
        super(BatchNorm2d, self).__init__(
            num_features,
            eps, momentum,
            affine, track_running_stats,
        )


class BatchNorm3d(_BatchNorm):
    r"""Apply the batch normalization over 4d input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = (1 - \text{momentum}) * x_{\text{running}}
                                   + \text{momentum} * x_{\text{batch}}

    See Also
    --------
    `torch.nn.functional.batch_norm(...)`_

    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
    ):
        r"""Create a ``BatchNorm3d`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.

        """
        super(BatchNorm3d, self).__init__(
            num_features,
            eps, momentum,
            affine, track_running_stats,
        )


class SyncBatchNorm(_BatchNorm):
    r"""Apply the sync batch normalization over input.
    `[Ioffe & Szegedy, 2015] <https://arxiv.org/abs/1502.03167>`_.

    The normalization is defined as:

    .. math:: y = \frac{x - \mathrm{E}[x]}
                       {\sqrt{\mathrm{Var}[x] + \epsilon}}
                  * \gamma + \beta

    The running average of statistics are calculated as:

    .. math:: x_{\text{running}} = (1 - \text{momentum}) * x_{\text{running}}
                                   + \text{momentum} * x_{\text{batch}}

    If :attr:`process_group` is ``None``,
    use the value of ``dragon.distributed.get_group(...)``.

    See Also
    --------
    `torch.nn.functional.sync_batch_norm(...)`_

    """

    def __init__(
        self,
        num_features,
        eps=1e-5,
        momentum=0.1,
        affine=True,
        track_running_stats=True,
        process_group=None,
    ):
        r"""Create a ``SyncBatchNorm`` module.

        Parameters
        ----------
        num_features : int
            The number of channels.
        eps : float, optional, default=1e-5
            The value to :math:`\epsilon`.
        momentum : float, optional, default=0.1
            The value to :math:`\text{momentum}`.
        affine : bool, optional, default=True
            ``True`` to apply an affine transformation.
        track_running_stats : bool, optional, default=True
            ``True`` to using stats when switching to ``eval``.
        process_group : ProcessGroup, optional
            The group for communication.

        """
        super(SyncBatchNorm, self).__init__(
            num_features, eps, momentum,
            affine, track_running_stats,
        )
        if process_group is None:
            process_group = distributed.get_group()
        self.process_group = process_group

    def forward(self, input):
        if self.training:
            return F.sync_batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=self.training,
                momentum=self._get_momentum(),
                eps=self.eps,
                process_group=self.process_group)
        else:
            return F.batch_norm(
                input,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=self.training,
                momentum=self._get_momentum(),
                eps=self.eps)

    @classmethod
    def convert_sync_batchnorm(cls, module, process_group=None):
        """Convert to sync batch normalization recursively.

        Parameters
        ----------
        module : dragon.vm.torch.nn.Module
            The module containing batch normalization.
        process_group : ProcessGroup, optional
            The group for communication.

        Returns
        -------
        dragon.vm.torch.nn.Module
            The output module.

        """
        module_output = module
        if isinstance(module, _BatchNorm):
            module_output = SyncBatchNorm(
                module.num_features,
                module.eps,
                module.momentum,
                module.affine,
                module.track_running_stats,
                process_group)
            if module.affine:
                module_output.weight = module.weight
                module_output.bias = module.bias
            module_output.running_mean = module.running_mean
            module_output.running_var = module.running_var
            module_output.num_batches_tracked = module.num_batches_tracked
        for name, child in module.named_children():
            module_output.add_module(
                name, cls.convert_sync_batchnorm(child, process_group))
        del module
        return module_output
