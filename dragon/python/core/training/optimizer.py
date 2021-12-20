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
"""Basic optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import itertools

import numpy

from dragon.core import distributed
from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.graph_lib import GraphLib
from dragon.core.framework import workspace


class Optimizer(object):
    """The base class of optimizers."""

    def __init__(self, **kwargs):
        """Create a ``Optimizer``."""
        self._name = workspace.get_workspace().create_handle('Optimizer')
        self._op_type = self.__class__.__name__
        self._process_group = distributed.get_group()
        self._hyper = {}
        self._set_hyper('grad_scale', kwargs.pop('grad_scale', 1))
        self._set_hyper('weight_decay', kwargs.pop('weight_decay', 0))
        self._set_hyper('clip_norm', kwargs.pop('clip_norm', 0))
        self._set_hyper('clip_value', kwargs.pop('clip_value', 0))
        if kwargs:
            raise ValueError('Unexpected arguments: ' + ','.join(v for v in kwargs))

    def apply_gradients(self, grads_and_vars):
        """Apply the gradients on variables.

        Parameters
        ----------
        grads_and_vars : Sequence[Sequence[dragon.Tensor]]
            The sequence of update pair.

        """
        # Create execution context for graph mode.
        if not context.executing_eagerly():
            return GraphLib.from_updates(grads_and_vars, self)

        # Separate variables by explicit weight decay.
        group_vars = collections.defaultdict(list)
        group_grads = collections.defaultdict(list)
        for grad, var in grads_and_vars:
            if grad is not None:
                weight_decay = getattr(var, '_weight_decay', None)
                if weight_decay is not None:
                    weight_decay = float(weight_decay)
                group_vars[weight_decay].append(var)
                group_grads[weight_decay].append(grad)

        # Reduce grads in the process group.
        process_group = distributed.get_group()
        if process_group is not None:
            grads = list(itertools.chain(*group_grads.values()))
            OpLib.execute('Collective', grads, outputs=grads,
                          operation='ALLREDUCE', reduction='MEAN',
                          **process_group.arguments)

        # Apply updates.
        for weight_decay, vars in group_vars.items():
            grads = group_grads[weight_decay]
            # Skip if grads are all missing.
            if len(grads) == 0:
                continue
            OpLib.execute(self._op_type, grads, outputs=vars,
                          name=self._name, weight_decay=weight_decay)

    def _set_hyper(self, name, value):
        """Set value to a hyper parameter."""
        if name not in self._hyper:
            default_ws = workspace.get_workspace()
            impl = default_ws.create_tensor(self._name + '/' + name)
            self._hyper[name] = impl
        value = numpy.array(float(value), 'float32')
        self._hyper[name].FromNumpy(value, False)

    def __getattr__(self, item):
        hyper = self.__dict__.get('_hyper')
        if item in hyper:
            return float(self._hyper[item].ToNumpy(False))
        return self.__dict__[item]

    def __setattr__(self, key, value):
        hyper = self.__dict__.get('_hyper')
        if hyper and key in hyper:
            value = numpy.array(float(value), 'float32')
            hyper[key].FromNumpy(value, False)
        else:
            object.__setattr__(self, key, value)
