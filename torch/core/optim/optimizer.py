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
#     <https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py>
#
# ------------------------------------------------------------
"""Basic optimizers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from dragon.core import distributed
from dragon.core.framework import workspace
from dragon.vm.torch.core.ops.distributed import functional as distributed_funcs
from dragon.vm.torch.core.ops.training import functional as training_funcs
from dragon.vm.torch.core.tensor import Tensor

# A simple parameter flag.
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

    # Store for the global unique handle
    _DEFAULT_UNIQUE_HANDLE_INDEX = 0

    def __init__(self, params, defaults):
        """Create a ``Optimizer``.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        defaults : dict
            The pre-defined default hyper-parameters.

        """
        self.defaults = defaults
        if isinstance(params, Tensor):
            raise TypeError('<params> should be a sequence of tensors.')
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError('Got an empty parameter list')
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)
        self._op_type = self.__class__.__name__ + 'Update'
        self._process_group = distributed.get_group()
        self._shared_args = {}

    def accumulate(self, momentum):
        """Accumulate the gradient of params.

        Call this method after each ``backward`` pass:

        ```python
        x = torch.ones(1, requires_grad=True)
        optimizer = torch.optim.SGD([x], lr=0.1)
        for epoch in range(2):
            for step in range(3):
                y = x + 1
                y.backward()
                # Note to zero the accumulation at the first step
                optimizer.accumulate(momentum=1 if step > 0 else 1)
            optimizer.step()
        print(x)  # 0.4
        ```

        Parameters
        ----------
        momentum : float, required
            The momentum to the accumulated value.

        """
        current_ws = workspace.get_workspace()
        for group in self.param_groups:
            group['_internal/grad_accum'] = True
            for param in group['params']:
                grad = self._steal_grad(current_ws, param)
                if grad is not None:
                    training_funcs.accumulate_grad(grad)

    def add_param_group(self, param_group):
        """Add a new param group into the optimizer.

        The ``param_group`` should be a dict containing
        the defaults optionally:

        ```python
        # A group redefined ``lr`` and ``weight_decay``
        param_group1 = {
            'params': [],
            'lr': 0.01,
            'weight_decay': 0.0001,
        }
        # A group inherits the defaults while using ``multiplier``
        param_group2 = {
            'params': [],
            'lr_mult': 1,
            'decay_mult': 1,
        }
        ```

        Parameters
        ----------
        param_group : dict
            The param group to add.

        """
        if not isinstance(param_group, dict):
            raise TypeError('Param group must be a dict.')

        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif isinstance(params, (set, dict)):
            raise TypeError('Parameters should be organized in a sequence.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not param.requires_grad:
                raise ValueError("Optimize a parameter that doesn't require grad.")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "Parameter group didn't specify a value of "
                    "required optimization parameter: " + name)
            else:
                param_group.setdefault(name, default)

        if 'name' not in param_group:
            Optimizer._DEFAULT_UNIQUE_HANDLE_INDEX += 1
            param_group['name'] = 'Optimizer_{}'.format(
                Optimizer._DEFAULT_UNIQUE_HANDLE_INDEX)

        if '_internal/grad_accum' not in param_group:
            param_group['_internal/grad_accum'] = False

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError('Some parameters appear in more than one parameter group.')

        self.param_groups.append(param_group)

    def step(self):
        """Perform one step update.

        Call this method after a ``backward`` pass:

        ```python
        x = torch.ones(1, 3, requires_grad=True)
        y = x + 1
        y.backward()
        optimizer.step()
        ```

        """
        current_ws = workspace.get_workspace()
        for group in self.param_groups:
            self._run_updates(current_ws, group)
            group['_internal/grad_accum'] = False

    def zero_grad(self, reset=False):
        """Set the gradient of params to zero.

        This method is not necessary usually, as we will overwrite
        the gradients in the next computation.

        However, if some gradients are not computed every time,
        remember to reset them before ``step(...)``:

        ```python
        m1 = torch.nn.Linear(3, 3)
        m2 = torch.nn.Linear(3, 3)
        x = torch.ones(1, 3, requires_grad=True)
        for i in range(10):
            x = m1(x)
            if i in (2, 4, 6):
                x += m2(x)
        optimizer.zero_grad(reset=True)
        x.backward()
        optimizer.step()
        ```

        Parameters
        ----------
        reset : bool, optional, default=False
            **True** to reset the memory instead of zeroing.

        """
        current_ws = workspace.get_workspace()
        for group in self.param_groups:
            for param in group['params']:
                grad = self._steal_grad(current_ws, param)
                if grad is not None:
                    current_ws.reset_tensor(grad) if reset else grad.zero_()

    def _run_updates(self, ws, group):
        """Run updates for the parameter group."""
        # Collect params and grads.
        params, grads = [], []
        grad_accum = group['_internal/grad_accum']
        for p in group['params']:
            g = self._steal_grad(ws, p, grad_accum)
            if g is not None:
                params.append(p)
                grads.append(g)

        # Reset the shared defaults.
        self._reset_defaults(ws, group)

        # Accumulate grads from the current process group.
        if self._process_group is not None:
            distributed_funcs.all_reduce(
                tensor=grads,
                op='MEAN',
                group=self._process_group,
            )

        # Apply the specific update.
        for p, g in zip(params, grads):
            training_funcs.update_param(
                p, g,
                op_type=self._op_type,
                op_handle=group['name'],
                lr_mult=group.get('lr_mult', 1),
                decay_mult=group.get('decay_mult', 1),
            )

    def _reset_defaults(self, ws, group):
        """Reset the defaults to backend."""
        template = '/share/hyper/%s/{}' % group['name']
        for name, value in group.items():
            if name in self._shared_args:
                ws.feed_tensor(
                    tensor=template.format(self._shared_args[name]),
                    value=value,
                    dtype='float32',
                    enforce_cpu=True,
                )

    @staticmethod
    def _steal_grad(ws, param, grad_accum=False):
        """Steal the grad from backend."""
        impl = ws.GetTensor(param.id + ('_grad[accum]' if grad_accum else '_grad'))
        if impl is not None:
            return Tensor(device=param.device, impl=impl)
        return None

    def __repr__(self):
        format_string = self.__class__.__name__ + ' ('
        for i, group in enumerate(self.param_groups):
            format_string += '\n'
            format_string += 'Parameter Group {0}\n'.format(i)
            for key in sorted(group.keys()):
                if key != 'params':
                    format_string += '    {0}: {1}\n'.format(key, group[key])
        format_string += ')'
        return format_string
