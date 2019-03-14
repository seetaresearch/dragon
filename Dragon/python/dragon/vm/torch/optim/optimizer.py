# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#      <https://github.com/pytorch/pytorch/blob/master/torch/optim/optimizer.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import dragon
from collections import defaultdict

from dragon.vm.torch.tensor import Tensor

from dragon.vm.torch.ops.builtin import (
    _accumulate, _allreduce, _update,
)


# A simple parameter flag
required = object()


class Optimizer(object):
    # Store the global unique slot index
    _DEFAULT_UNIQUE_SLOT_ID = 0

    def __init__(self, params, defaults):
        self.defaults = defaults
        if isinstance(params, Tensor):
            raise TypeError("params argument given to the optimizer should be "
                            "an iterable of Variables or dicts, but got " +
                            str(type(params)))
        self.state = defaultdict(dict)
        self.param_groups = []
        param_groups = list(params)
        if len(param_groups) == 0:
            raise ValueError("optimizer got an empty parameter list")
        if not isinstance(param_groups[0], dict):
            param_groups = [{'params': param_groups}]
        for param_group in param_groups:
            self.add_param_group(param_group)
        self._update_type = None
        self._allow_parallel = False
        if dragon.mpi.Is_Init():
            local_rank, _ = dragon.mpi.AllowParallel()
            if local_rank != -1: self._allow_parallel = True
        self._mutable_parameters = {}

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

    def feed_parameters(self, group):
        template = group['slot'] + '/{}'
        for k, v in group.items():
            if k in self._mutable_parameters:
                dragon.workspace.FeedTensor(
                    template.format(self._mutable_parameters[k]),
                        v, dtype='float32', force_cpu=True)

    def _get_grad(self, param, accumulating=False):
        grad_name = param.name + (
            '_grad[acc]' if accumulating
                else '_grad')
        if dragon.workspace.HasTensor(grad_name):
            return Tensor(
                name=grad_name,
                    own_storage=False,
                        device=param.device)
        return None

    def _run_update_ops(self, group):
        """Generate & Run UpdateOps.

        Parameters
        ----------
        group : dict
            The param group.

        Returns
        -------
        None

        """
        # Collect params and grads
        params, grads = [], []
        for p in group['params']:
            g = self._get_grad(
                p, p.__accumulating__)
            if g is not None:
                params.append(p)
                grads.append(g)

        # Feed optimizer parameters to workspace
        self.feed_parameters(group)

        # Run a all-reduce op to accumulate grads if necessary
        if self._allow_parallel: _allreduce(grads)

        # Run regular update ops
        for p, g in zip(params, grads):
            _update(
                p, g,
                op_type=self._update_type,
                slot=group['slot'],
                lr_mult=group.get('lr_mult', 1.0),
                decay_mult=group.get('decay_mult', 1.0),
            )

    def zero_grad(self):
        """Set all gradients to zeros."""
        for group in self.param_groups:
            for p in group['params']:
                g = self._get_grad(
                    p, p.__accumulating__)
                p.__accumulating__ = False
                if g is not None:
                    g.zero_()

    def accumulate_grad(self):
        """Accumulate all gradients."""
        grads = []
        for group in self.param_groups:
            for p in group['params']:
                g = self._get_grad(p)
                if g is not None:
                    grads.append(g)
                    p.__accumulating__ = True
        _accumulate(grads)

    def step(self, closure=None):
        """Perform one step update.

        Returns
        -------
        None

        """
        for group in self.param_groups:
            self._run_update_ops(group)

    def add_param_group(self, param_group):
        """Add a new param group into the optimizer.

        param_group : dict
            The param group.

        Returns
        -------
        None

        """
        assert isinstance(param_group, dict), "Param group must be a dict."

        params = param_group['params']

        if isinstance(params, Tensor):
            param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('Optimizer parameters need to be organized in ordered collections,'
                            '\nbut the ordering of tensors in sets will change between runs.'
                            '\nPlease use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not param.requires_grad:
                raise ValueError("Optimizing a Parameter({}) that "
                                 "doesn't require gradients".format(param.name))

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("Parameter group didn't specify a value of "
                                 "required optimization parameter: " + name)
            else:
                param_group.setdefault(name, default)

        if 'slot' not in param_group:
            Optimizer._DEFAULT_UNIQUE_SLOT_ID += 1
            param_group['slot'] = 'Optimizer/Slot:{}'.format(
                Optimizer._DEFAULT_UNIQUE_SLOT_ID)

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)