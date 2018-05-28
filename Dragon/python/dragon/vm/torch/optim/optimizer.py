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

import warnings
from collections import defaultdict

import numpy as np
import dragon as dg

from dragon.vm.torch.tensor import Tensor
from dragon.vm.torch.ops.update import _allreduce, _update


_OPTIMIZER_GROUP_UID = 0
required = object()


def _get_group_slot():
    global _OPTIMIZER_GROUP_UID
    _OPTIMIZER_GROUP_UID += 1
    return 'optimizer_group:{}'.format(_OPTIMIZER_GROUP_UID - 1)


class Optimizer(object):
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
        param_temp = group['slot'] + '/{}'
        for k, v in group.items():
            if k in self._mutable_parameters:
                # convert all defaults as float32 for convenience
                dg.workspace.FeedTensor(param_temp.format(
                    self._mutable_parameters[k]),
                        np.array([v], dtype=np.float32))

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
        params = []; grads = []
        for p in group['params']:
            g_name = p.name + '_grad'
            if not dg.workspace.HasTensor(g_name): continue
            g = Tensor(dg_tensor=g_name)
            g._own_storage = False; g._ctx = p._ctx
            params.append(p); grads.append(g)

        # Feed optimizer parameters to workspace
        self.feed_parameters(group)

        # Run a all-reduce op to accumulate grads if necessary
        _allreduce(grads)

        # Run regular update ops
        for p, g in zip(params, grads):
            _update(p, g, op_type=self._update_type,
                slot=group['slot'],
                lr_mult=group.get('lr_mult', 1.0),
                decay_mult=group.get('decay_mult', 1.0))

    def zero_grad(self):
        """Set all gradients to zeros.

        Deprecated, DO NOT call it.

        """
        warnings.warn("\nPyTorch@Dragon will automatically zeros the grad "
                      "after each step. \nYou need not call this function, "
                      "IT'S TOO SLOW.", stacklevel=2)
        for group in self.param_groups:
            for p in group['params']:
                g = p.grad
                if g is not None: g.zero_()

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
        assert isinstance(param_group, dict), "param group must be a dict"

        params = param_group['params']
        if isinstance(params, Tensor): param_group['params'] = [params]
        elif isinstance(params, set):
            raise TypeError('optimizer parameters need to be organized in ordered collections, but '
                            'the ordering of tensors in sets will change between runs. Please use a list instead.')
        else:
            param_group['params'] = list(params)

        for param in param_group['params']:
            if not param.requires_grad:
                print(param.name)
                raise ValueError("optimizing a parameter that doesn't require gradients")

        for name, default in self.defaults.items():
            if default is required and name not in param_group:
                raise ValueError("parameter group didn't specify a value of required optimization parameter " +
                                 name)
            else:
                param_group.setdefault(name, default)

        param_group['slot'] = _get_group_slot()

        param_set = set()
        for group in self.param_groups:
            param_set.update(set(group['params']))

        if not param_set.isdisjoint(set(param_group['params'])):
            raise ValueError("some parameters appear in more than one parameter group")

        self.param_groups.append(param_group)