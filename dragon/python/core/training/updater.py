# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

"""Define the base updater class.

We dubbed it as ``Updater``, because ``Optimizer``
has already been abused by many deep learning frameworks.

"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core import distributed
from dragon.core.eager import context
from dragon.core.framework import workspace
from dragon.core.ops import distributed_ops_lib
from dragon.core.ops import training_ops_lib


class Updater(object):
    """The base class of updaters."""

    # Store the global unique slot index.
    _DEFAULT_UNIQUE_SLOT_ID = 0

    def __init__(
        self,
        scale_gradient=1.,
        clip_gradient=-1.,
        l2_decay=-1.,
        name=None,
    ):
        """Create an ``Updater``.

        Parameters
        ----------
        scale_gradient : float, optional, default=1.
            The factor to scale gradients.
        clip_gradient : float, optional, default=-1.
            The norm thresh to clip gradients.
        l2_decay : float, optional, default=-1.
            The l2 decay factor.
        name : str, optional
            The optional name for buffers.

        """
        self._defaults = {
            'scale_gradient': scale_gradient,
            'clip_gradient': clip_gradient,
            'l2_decay': l2_decay,
        }
        self._param_group = []
        if name:
            self._slot = name
        else:
            Updater._DEFAULT_UNIQUE_SLOT_ID += 1
            self._slot = 'Updater/Slot:{}'.format(
                Updater._DEFAULT_UNIQUE_SLOT_ID)
        self._op_type = None
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
            The multiplier on learning rate.
        decay_mult : number, optional
            The multiplier on weight decay.

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
        self._op_type = self.__class__.__name__ + 'Update'
        for k, v in self._defaults.items():
            workspace.feed_tensor(
                self._slot + "/" + k, v,
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
                slot=self._slot,
                op_type=self._op_type,
                lr_mult=float(lr_mult) if lr_mult is not None else 1.,
                decay_mult=float(decay_mult) if decay_mult is not None else 1.,
            ).apply(grad, param)

    def __getattr__(self, item):
        defaults = self.__dict__.get('_defaults')
        if item in defaults:
            return workspace.fetch_tensor(
                self._slot + '/' + item)
        return self.__dict__[item]

    def __setattr__(self, key, value):
        defaults = self.__dict__.get('_defaults')
        if defaults is not None and key in defaults:
            workspace.feed_tensor(
                self._slot + '/' + key, value,
                dtype='float32', enforce_cpu=True,
            )
        else:
            object.__setattr__(self, key, value)
