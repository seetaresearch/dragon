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
"""Basic optimizers."""

import collections
import itertools

import numpy

from dragon.core.distributed import backend as dist_backend
from dragon.core.autograph import context
from dragon.core.autograph.op_lib import OpLib
from dragon.core.autograph.graph_lib import GraphLib
from dragon.core.framework import workspace


class Optimizer(object):
    """The base class of optimizers."""

    def __init__(self, **kwargs):
        """Create a ``Optimizer``."""
        self._name = workspace.get_workspace().create_handle("Optimizer")
        self._op_type = self.__class__.__name__
        self._process_group = dist_backend.get_group()
        self._hyper_dict = {}
        self._set_hyper("grad_scale", kwargs.pop("grad_scale", 1))
        self._set_hyper("weight_decay", kwargs.pop("weight_decay", 0))
        self._set_hyper("clip_norm", kwargs.pop("clip_norm", 0))
        self._set_hyper("clip_value", kwargs.pop("clip_value", 0))
        if kwargs:
            raise ValueError("Unexpected arguments: " + ",".join(v for v in kwargs))

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
                weight_decay = getattr(var, "_weight_decay", None)
                if weight_decay is not None:
                    weight_decay = float(weight_decay)
                group_vars[weight_decay].append(var)
                group_grads[weight_decay].append(grad)

        # Reduce grads in the distribution group.
        dist_group = dist_backend.get_group()
        if dist_group is not None:
            grads = list(itertools.chain(*group_grads.values()))
            OpLib.execute(
                "Collective",
                grads,
                outputs=grads,
                operation="ALLREDUCE",
                reduction="MEAN",
                **dist_group.arguments
            )

        # Apply update.
        for weight_decay, vars in group_vars.items():
            grads = group_grads[weight_decay]
            # Skip if grads are all missing.
            if len(grads) == 0:
                continue
            OpLib.execute(
                self._op_type,
                grads,
                outputs=vars,
                name=self._name,
                weight_decay=weight_decay,
            )

    def _set_hyper(self, hyper_name, new_value):
        """Set value to a hyper parameter."""
        if hyper_name not in self._hyper_dict:
            execute_ws = workspace.get_workspace()
            impl = execute_ws.create_tensor(self._name + "/" + hyper_name)
            self._hyper_dict[hyper_name] = impl
        new_value = numpy.array(float(new_value), "float32")
        self._hyper_dict[hyper_name].FromNumpy(new_value, False)

    def __getattr__(self, item):
        hyper_dict = self.__dict__.get("_hyper_dict")
        if hyper_dict and item in hyper_dict:
            return float(hyper_dict[item].ToNumpy(False))
        return self.__dict__[item]

    def __setattr__(self, key, value):
        hyper_dict = self.__dict__.get("_hyper_dict")
        if hyper_dict and key in hyper_dict:
            new_value = numpy.array(float(value), "float32")
            hyper_dict[key].FromNumpy(new_value, False)
        else:
            object.__setattr__(self, key, value)
