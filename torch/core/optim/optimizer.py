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
"""Basic optimizer."""

import collections
import numpy

from dragon.core.distributed import backend as dist_backend
from dragon.core.framework import workspace
from dragon.vm.torch.core.autograd.function import Function
from dragon.vm.torch.core.tensor import Tensor

required = object()


class Optimizer(object):
    """The base class of optimizers.

    Inherit this class to design a new optimizer:

    ```python
    class MyOptimizer(torch.optim.Optimizer):
        def __init__(params, hp1, hp2):
            defaults = dict(hp1=hp1, hp2=hp2)
            super(MyOptimizer, self).__init__(params, defaults)
    ```

    """

    def __init__(self, params, defaults, **kwargs):
        """Create a ``Optimizer``.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        defaults : dict
            The default values of optimization.

        """
        if isinstance(params, Tensor):
            raise TypeError("<params> should be a sequence of tensors.")
        defaults["grad_scale"] = kwargs.pop("grad_scale", 1)
        defaults["clip_norm"] = kwargs.pop("clip_norm", 0)
        defaults["clip_value"] = kwargs.pop("clip_value", 0)
        self.bucket_size = kwargs.pop("bucket_size", 536870912)
        if kwargs:
            raise ValueError("Unexpected arguments: " + ",".join(v for v in kwargs))
        self.defaults = defaults
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("<params> is an empty sequence.")
        if not isinstance(param_groups[0], dict):
            param_groups = [{"params": param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)
        self._op_type = self.__class__.__name__
        self._hyper_dict = dict((k, dict()) for k in self.defaults.keys())
        self._sums_grad = False
        self._handle_custom_step = False

    def add_param_group(self, param_group):
        """Add a new param group into the optimizer.

        attr:`param_group` is a dict containing the defaults:

        ```python
        # A group defined ``lr`` and ``weight_decay``
        param_group = {'params': [], 'lr': 0.01, 'weight_decay': 0.0001}
        ```

        Parameters
        ----------
        param_group : dict
            The param group to add.

        """
        if not isinstance(param_group, dict):
            raise TypeError("Param group must be a dict.")
        params = param_group["params"]
        if isinstance(params, Tensor):
            param_group["params"] = [params]
        elif isinstance(params, (set, dict)):
            raise TypeError("Parameters should be organized in a sequence.")
        else:
            param_group["params"] = list(params)
        for param in param_group["params"]:
            if not isinstance(param, Tensor):
                raise TypeError("Can only optimize tensors, but one is " + str(type(param)))
            if not param.is_leaf:
                raise ValueError("Optimize a non-leaf Tensor.")
        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("Parameter group didn't specify a required value: {}".format(name))
            else:
                param_group.setdefault(name, default)
        if "name" not in param_group:
            default_ws = workspace.get_workspace()
            param_group["name"] = default_ws.create_handle("Optimizer")
        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group["params"]))
        if not param_set.isdisjoint(set(param_group["params"])):
            raise ValueError("Some parameters appear in more than one parameter group.")
        self.param_groups.append(param_group)

    def step(self):
        """Update all parameter groups using gradients.

        Call this method after a ``backward`` pass:

        ```python
        x = torch.ones(1, 3, requires_grad=True)
        y = x + 1
        y.backward()
        optimizer.step()
        ```

        """
        execute_ws = workspace.get_workspace()
        dist_group = dist_backend.get_group()
        params_all, grads_all = [], []
        reduce_grads_all = collections.defaultdict(list)
        for group in self.param_groups:
            group.setdefault("dist_size", dist_group.size) if dist_group else None
            params_with_grad, grads = [], []
            for p in group["params"]:
                g = self._get_grad(execute_ws, p, self._sums_grad)
                if g is not None:
                    grads.append(g)
                    params_with_grad.append(p)
                    reduce_grads_all[g.dtype].append(g) if dist_group else None
            grads_all.append(grads)
            params_all.append(params_with_grad)
        for grads in reduce_grads_all.values():
            Function.apply(
                "Collective",
                grads[0].device,
                grads,
                outputs=grads,
                operation="ALLREDUCE",
                reduction="SUM",
                bucket_size=self.bucket_size,
                **dist_group.arguments,
            )
        self._sums_grad = False
        if self._handle_custom_step:
            return params_all, grads_all
        for idx, group in enumerate(self.param_groups):
            self._update_group(group, params_all[idx], grads_all[idx])

    def sum_grad(self):
        """Sum the gradients of all parameters.

        Call this method after each ``backward`` pass:

        ```python
        x = torch.ones(1, requires_grad=True)
        optimizer = torch.optim.SGD([x], lr=0.1)
        for epoch in range(2):
            for step in range(3):
                y = x + 1
                y.backward()
                optimizer.sum_grad()
            optimizer.step()
        print(x)  # 0.4
        ```

        """
        execute_ws = workspace.get_workspace()
        grads_all, sum_grads_all = [], []
        for group in self.param_groups:
            grads, sum_grads = [], []
            for p in group["params"]:
                g = self._get_grad(execute_ws, p)
                if g is not None:
                    grads.append(g)
                    sum_grads.append(g.id + "_sum")
            grads_all.append(grads)
            sum_grads_all.append(sum_grads)
        for grads, sum_grads in zip(grads_all, sum_grads_all):
            Function.apply(
                "Axpby",
                grads[0].device,
                grads,
                outputs=sum_grads,
                alpha=1.0,
                beta=1.0 if self._sums_grad else 0.0,
            )
        self._sums_grad = True

    def zero_grad(self, set_to_none=False):
        """Set the gradients of all parameters to zero.

        This method is not necessary usually, as we will overwrite
        the gradients in the next computation.

        However, if some gradients are not computed every time,
        remember to set them to none before ``step(...)``:

        ```python
        m1 = torch.nn.Linear(3, 3)
        m2 = torch.nn.Linear(3, 3)
        x = torch.ones(1, 3, requires_grad=True)
        for i in range(10):
            x = m1(x)
            if i in (2, 4, 6):
                x += m2(x)
        optimizer.zero_grad(set_to_none=True)
        x.backward()
        optimizer.step()
        ```

        Parameters
        ----------
        set_to_none : bool, optional, default=False
            Whether to remove the gradients instead of zeroing.

        """
        execute_ws = workspace.get_workspace()
        for group in self.param_groups:
            for p in group["params"]:
                g = self._get_grad(execute_ws, p)
                if g is not None:
                    _ = g._impl.Reset() if set_to_none else g.zero_()

    @staticmethod
    def _get_grad(execute_ws, param, summed=False):
        """Return the grad of a parameter."""
        impl = execute_ws.get_tensor(param.id + "_grad_sum") if summed else None
        impl = execute_ws.get_tensor(param.id + "_grad") if impl is None else impl
        return Tensor(device=param.device, impl=impl) if impl else None

    def _update_group(self, group, params, grads):
        """Update parameters for the group."""
        # Skip if grads are all missing.
        if len(params) == 0:
            return
        execute_ws = workspace.get_workspace()
        # Update hyper tensors with current value.
        for hyper_name in self._hyper_dict.keys():
            group_dict = self._hyper_dict[hyper_name]
            group_name, new_value = group["name"], group[hyper_name]
            if group_name not in group_dict:
                impl_name = group_name + "/" + hyper_name
                group_dict[group_name] = execute_ws.create_tensor(impl_name)
            if hyper_name == "grad_scale":
                new_value = new_value / group.pop("dist_size", 1)
            group_dict[group_name].FromNumpy(numpy.array(new_value, "float32"), False)
        # Apply update.
        Function.apply(
            self._op_type,
            params[0].device,
            grads,
            outputs=params,
            name=group["name"],
            weight_decay=None,
        )

    def __repr__(self):
        format_string = self.__class__.__name__ + " ("
        for i, group in enumerate(self.param_groups):
            format_string += "\n"
            format_string += "Parameter Group {0}\n".format(i)
            for key in sorted(group.keys()):
                if key != "params":
                    format_string += "    {0}: {1}\n".format(key, group[key])
        format_string += ")"
        return format_string
