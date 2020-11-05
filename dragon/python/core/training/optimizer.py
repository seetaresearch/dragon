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
"""The optimizer to update parameters."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core import distributed
from dragon.core.eager import context
from dragon.core.framework import workspace
from dragon.core.ops import distributed_ops_lib
from dragon.core.ops import training_ops_lib


class Optimizer(object):
    """The base class of optimizers."""

    # Store for the global unique handle
    _DEFAULT_UNIQUE_HANDLE_INDEX = 0

    def __init__(
        self,
        scale=1,
        clip_norm=0,
        weight_decay=0,
        name=None,
    ):
        """Create a ``Optimizer``.

        Parameters
        ----------
        scale : float, optional, default=1
            The scaling factor to gradient.
        clip_norm : float, optional, default=0
            The maximum L2 norm to clip gradient.
        weight_decay : float, optional, default=0
            The L2 penalty factor to weight.
        name : str, optional
            The optional name for shared slots.

        """
        self._defaults = {
            'scale': float(scale),
            'clip_norm': float(clip_norm),
            'weight_decay': float(weight_decay),
        }
        self._param_group = []
        if name:
            self._op_handle = name
        else:
            Optimizer. _DEFAULT_UNIQUE_HANDLE_INDEX += 1
            self._op_handle = 'Optimizer_{}'.format(
                Optimizer. _DEFAULT_UNIQUE_HANDLE_INDEX)
        self._op_type = self.__class__.__name__ + 'Update'
        self._process_group = distributed.get_group()
        self._extra_kwargs = {}

    def apply_gradients(
        self,
        values_and_grads,
        lr_mult=None,
        decay_mult=None,
    ):
        """Apply the gradients on values.

        Parameters
        ----------
        values_and_grads : Sequence[Sequence[dragon.Tensor]]
            The values and grads.
        lr_mult : number, optional
            The multiplier to learning rate.
        decay_mult : number, optional
            The multiplier to weight decay.

        """
        if context.executing_eagerly():
            # Filter value whose grad is missing.
            values, grads = [], []
            for v, g in values_and_grads:
                if g is not None:
                    values.append(v)
                    grads.append(g)
            # Accumulate grads from the current process group.
            if self._process_group is not None:
                distributed_ops_lib.Collective \
                    .instantiate(
                        operation='MEAN',
                        communication='ALLREDUCE',
                        group=self._process_group,
                    ).apply(grads)
            # Apply the updates.
            for v, g in zip(values, grads):
                self._run_update(v, g, lr_mult, decay_mult)
        else:
            # Store for the lazy compilation.
            for v, g in values_and_grads:
                self._add_update(v, g, lr_mult, decay_mult)
        return self

    def _init_set_defaults(self, extra=None):
        """Initialize the defaults into current workspace."""
        if extra is not None:
            self._defaults = dict(self._defaults, **extra)
        for k, v in self._defaults.items():
            workspace.get_workspace().feed_tensor(
                '/share/hyper/%s/%s' % (self._op_handle, k), v,
                dtype='float32', enforce_cpu=True,
            )

    def _add_update(self, param, grad, lr_mult=None, decay_mult=None):
        """Add a symbolic operator for updating."""
        pair = (v.id if hasattr(v, 'id') else v for v in (param, grad))
        self._param_group.append(
            (pair, {
                'lr_mult': float(lr_mult) if lr_mult is not None else 1.,
                'decay_mult': float(decay_mult) if decay_mult is not None else 1.,
            })
        )

    def _run_update(self, param, grad, lr_mult=None, decay_mult=None):
        """Run an eager operation for updating."""
        return training_ops_lib.ParamUpdate \
            .instantiate(
                op_type=self._op_type,
                op_handle=self._op_handle,
                lr_mult=float(lr_mult) if lr_mult is not None else 1.,
                decay_mult=float(decay_mult) if decay_mult is not None else 1.,
            ).apply(grad, param)

    def __getattr__(self, item):
        """
        Return the attribute of an item.

        Args:
            self: (todo): write your description
            item: (str): write your description
        """
        defaults = self.__dict__.get('_defaults')
        if item in defaults:
            return workspace.get_workspace().fetch_tensor(
                '/share/hyper/%s/%s' % (self._op_handle, item))
        return self.__dict__[item]

    def __setattr__(self, key, value):
        """
        Set a key on the value.

        Args:
            self: (todo): write your description
            key: (str): write your description
            value: (todo): write your description
        """
        defaults = self.__dict__.get('_defaults')
        if defaults is not None and key in defaults:
            workspace.get_workspace().feed_tensor(
                '/share/hyper/%s/%s' % (self._op_handle, key), value,
                dtype='float32', enforce_cpu=True)
        else:
            object.__setattr__(self, key, value)
