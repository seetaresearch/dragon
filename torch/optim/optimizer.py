# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#    <https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from collections import defaultdict

from dragon.core import distributed
from dragon.core.framework import workspace
from dragon.vm.torch.ops.distributed import functional as distributed_funcs
from dragon.vm.torch.ops.training import functional as training_funcs
from dragon.vm.torch.tensor import Tensor

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
        defaults : Dict
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

    def accumulate_grad(self):
        """Accumulate all gradients.

        Call this method after a ``backward`` pass:

        ```python
        x = torch.ones(1, 3, requires_grad=True)
        for i in range(10):
            y = x + 1
            y.backward()
            optimizer.accumulate_grad()
        optimizer.step()
        ```

        """
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                g = self._steal_grad(p)
                if g is not None:
                    grads.append(g)
                    p.__accumulating__ = True
        training_funcs.grad_accumulate(grads)

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
        param_group : Dict
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
                raise ValueError(
                    "Optimizing a parameter that "
                    "doesn't require gradients."
                )

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError(
                    "Parameter group didn't specify a value of "
                    "required optimization parameter: " + name
                )
            else:
                param_group.setdefault(name, default)

        if 'name' not in param_group:
            Optimizer._DEFAULT_UNIQUE_HANDLE_INDEX += 1
            param_group['name'] = 'Optimizer_{}'.format(
                Optimizer._DEFAULT_UNIQUE_HANDLE_INDEX)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError('Some parameters appear in '
                             'more than one parameter group.')

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
        for group in self.param_groups:
            self._run_updates(group)

    def zero_grad(self, reset=False):
        """Set all gradients to zeros.

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
        ```

        Parameters
        ----------
        reset : bool, optional, default=False
            **True** to reset the memory instead of zeroing.

        """
        for group in self.param_groups:
            for p in group['params']:
                g = self._steal_grad(p, p.__accumulating__)
                p.__accumulating__ = False
                if g is not None:
                    if reset:
                        workspace.reset_tensor(g)
                    else:
                        g.zero_()

    def _init_set_defaults(self, group):
        """Initialize the defaults into current workspace."""
        template = '/share/hyper/%s/{}' % group['name']
        for k, v in group.items():
            if k in self._shared_args:
                workspace.feed_tensor(
                    template.format(self._shared_args[k]),
                    v, dtype='float32', enforce_cpu=True)

    def _run_updates(self, group):
        """Run updates for the parameter group."""
        # Collect params and grads.
        params, grads = [], []
        for p in group['params']:
            g = self._steal_grad(p, p.__accumulating__)
            if g is not None:
                params.append(p)
                grads.append(g)

        # Reset the shared defaults.
        self._init_set_defaults(group)

        # Accumulate grads from the current process group.
        if self._process_group is not None:
            distributed_funcs.all_reduce(
                tensor=grads,
                op='MEAN',
                group=self._process_group,
            )

        # Apply the specific update.
        for p, g in zip(params, grads):
            training_funcs.param_update(
                p, g,
                op_type=self._op_type,
                op_handle=group['name'],
                lr_mult=group.get('lr_mult', 1),
                decay_mult=group.get('decay_mult', 1),
            )

    @staticmethod
    def _steal_grad(param, accumulating=False):
        """Steal the grad tensor if existing."""
        grad_id = param.id + ('_grad[acc]' if accumulating else '_grad')
        if workspace.has_tensor(grad_id):
            return Tensor(
                id=grad_id,
                own_storage=False,
                device=param.device,
            )
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
