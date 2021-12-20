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

import collections

import numpy

from dragon.core import distributed
from dragon.core.framework import workspace
from dragon.vm.torch.core.autograd.function import Function
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

    def __init__(self, params, defaults, **kwargs):
        """Create a ``Optimizer``.

        Parameters
        ----------
        params : Sequence[dragon.vm.torch.nn.Parameter]
            The parameters to optimize.
        defaults : dict
            The pre-defined default hyper-parameters.

        """
        if isinstance(params, Tensor):
            raise TypeError('<params> should be a sequence of tensors.')
        defaults.update({'grad_scale': kwargs.pop('grad_scale', 1),
                         'clip_norm': kwargs.pop('clip_norm', 0),
                         'clip_value': kwargs.pop('clip_value', 0)})
        if kwargs:
            raise ValueError('Unexpected arguments: ' + ','.join(v for v in kwargs))
        self.defaults = defaults
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError('<params> is an empty sequence.')
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)
        self._op_type = self.__class__.__name__
        self._hyper = dict((k, [k, collections.defaultdict(str)])
                           for k in self.defaults.keys())
        self._sums_grad = False

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
            raise TypeError('Param group must be a dict.')

        params = param_group['params']
        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif isinstance(params, (set, dict)):
            raise TypeError('Parameters should be organized in a sequence.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not isinstance(param, Tensor):
                raise TypeError('Optimizer can only optimize Tensors, '
                                "but one of the params is " + str(type(param)))
            if not param.is_leaf:
                raise ValueError("Optimize a non-leaf Tensor.")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "Parameter group didn't specify a value of "
                    "required optimization parameter: " + name)
            else:
                param_group.setdefault(name, default)

        if 'name' not in param_group:
            default_ws = workspace.get_workspace()
            param_group['name'] = default_ws.create_handle('Optimizer')

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))
        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError('Some parameters appear in more than one parameter group.')

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
        for group in self.param_groups:
            self._update_group(group)
        self._sums_grad = False

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
        current_ws = workspace.get_workspace()
        for group in self.param_groups:
            grads, sum_grads = [], []
            for param in group['params']:
                grad = self._get_grad(current_ws, param)
                if grad is not None:
                    grads.append(grad)
                    sum_grads.append(grad.id + '_sum')
            Function.apply(
                'Axpby', grads[0].device,
                grads, outputs=sum_grads,
                alpha=1., beta=1. if self._sums_grad else 0.)
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
            for param in group['params']:
                grad = self._get_grad(execute_ws, param)
                if grad is not None:
                    if set_to_none:
                        grad._impl.Reset()
                    else:
                        grad.zero_()

    def _update_group(self, group):
        """Update parameters for the group."""
        execute_ws = workspace.get_workspace()

        # Collect params and grads.
        params_with_grad, grads = [], []
        for p in group['params']:
            g = self._get_grad(execute_ws, p, self._sums_grad)
            if g is not None:
                params_with_grad.append(p)
                grads.append(g)

        # Skip if grads are all missing.
        if len(params_with_grad) == 0:
            return

        # Update hyper from group values.
        for name in self._hyper.keys():
            group_name = group['name']
            impl_name, group_dict = self._hyper[name]
            if group_name not in group_dict:
                impl_name = group_name + '/' + impl_name
                group_dict[group_name] = execute_ws.create_tensor(impl_name)
            impl = group_dict[group_name]
            impl.FromNumpy(numpy.array(group[name], 'float32'), False)

        # Reduce grads in the process group.
        process_group = distributed.get_group()
        if process_group is not None:
            Function.apply('Collective', grads[0].device, grads,
                           outputs=grads, operation='ALLREDUCE',
                           reduction='MEAN', **process_group.arguments)

        # Apply updates.
        Function.apply(self._op_type, params_with_grad[0].device, grads,
                       outputs=params_with_grad, name=group['name'],
                       weight_decay=None)

    @staticmethod
    def _get_grad(execute_ws, param, summed=False):
        """Return the grad of a parameter."""
        grad_impl = execute_ws.get_tensor(
            param.id + ('_grad_sum' if summed else '_grad'))
        if grad_impl:
            return Tensor(device=param.device, impl=grad_impl)
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
