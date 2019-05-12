# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import updaters as _updaters
from dragon.core import workspace as _workspace
from dragon.core.tensor import Tensor as _Tensor

from dragon.vm.tensorflow.framework import ops
from dragon.vm.tensorflow.ops import variables
from dragon.vm.tensorflow.ops.gradients_impl import gradients


class Optimizer(object):
    def __init__(self, use_locking, name):
        if not name:
            raise ValueError('Must specify the optimizer name.')
        self._use_locking = use_locking
        self._name = name
        self._slots = {}
        # Store the losses from gradients
        self._targets = []
        # Store the external global step
        self._global_step = None
        # Kept for building dragon updater
        self.updater = self.train = self.update = None

    def _set_dynamic_lr(self, learning_rate):
        if isinstance(learning_rate, _Tensor):
            self._targets.append(learning_rate)
            internal_lr = self.updater._slot + '/base_lr'
            _workspace.SetTensorAlias(learning_rate, internal_lr)
            self.updater.base_lr = float(learning_rate.get_value())

    def _inc_global_step(self):
        if self._global_step is not None:
            v = self._global_step.get_value() + 1
            _workspace.FeedTensor(self._global_step, v, True)

    def get_name(self):
        return self._name

    def minimize(self, loss, global_step=None, var_list=None, **kwargs):
        self._global_step = global_step
        grads_and_vars = self.compute_gradients(loss, var_list)
        return self.apply_gradients(grads_and_vars, global_step=global_step)

    def compute_gradients(self, loss, var_list=None, **kwargs):
        if var_list is None:
            var_list = variables.trainable_variables() + \
                ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        grads = gradients(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, **kwargs):
        self._global_step = global_step
        grads_and_vars = list(grads_and_vars)

        # Firstly, we should extract the potential decays
        l2_decays = []
        for grad, var in grads_and_vars:
            if hasattr(var, '__regularizer__'):
                if var.__regularizer__ and \
                        var.__regularizer__.l2 > 0:
                    l2_decays.append(var.__regularizer__.l2)

        # Find the base decay factor
        self.updater._defaults['l2_decay'] = \
            base_l2_decay = min(l2_decays) \
                if len(l2_decays) > 0 else - 1.0

        # Add to targets
        targets = set()
        for grad, var in grads_and_vars:
            decay_multiplier = 0.
            if hasattr(var, '__regularizer__'):
                if var.__regularizer__ and \
                        var.__regularizer__.l2 > 0:
                    decay_multiplier = \
                        var.__regularizer__.l2 / base_l2_decay
            self.updater.append((var, grad), decay_mult=decay_multiplier)
            targets.update(var.gradient.cost)
        self._targets.extend(list(targets))
        return self


class GradientDescentOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate,
        use_locking=False,
        name='GradientDescent',
    ):
        super(GradientDescentOptimizer, self).__init__(use_locking, name)
        self.updater = _updaters.SGDUpdater(learning_rate, 0.)
        self._set_dynamic_lr(learning_rate)


class MomentumOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate,
        momentum,
        use_locking=False,
        name='Momentum',
        use_nesterov=False,
    ):
        super(MomentumOptimizer, self).__init__(use_locking, name)
        if not use_nesterov:
            self.updater = _updaters.SGDUpdater(learning_rate, momentum)
        else:
            self.updater = _updaters.NesterovUpdater(learning_rate, momentum)
        self._set_dynamic_lr(learning_rate)


class AdamOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate=0.001,
        beta1=0.9,
        beta2=0.999,
        epsilon=1e-8,
        use_locking=False,
        name='Adam',
    ):
        super(AdamOptimizer, self).__init__(use_locking, name)
        self.updater = _updaters.AdamUpdater(
            learning_rate, beta1, beta2, epsilon)
        self._set_dynamic_lr(learning_rate)


class RMSPropOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate,
        decay=0.9,
        momentum=0.0,
        epsilon=1e-10,
        use_locking=False,
        centered=False,
        name='RMSProp',
    ):
        super(RMSPropOptimizer, self).__init__(use_locking, name)
        if momentum > 0.:
            self.updater = _updaters.AdamUpdater(
                learning_rate, momentum, decay, epsilon)
        else:
            self.updater = _updaters.RMSPropUpdater(
                learning_rate, decay, epsilon)
        self._set_dynamic_lr(learning_rate)