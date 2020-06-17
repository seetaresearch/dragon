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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon import updaters
from dragon.core.framework import types
from dragon.core.framework import workspace
from dragon.vm.tensorflow.core.framework import ops
from dragon.vm.tensorflow.core.ops import variables
from dragon.vm.tensorflow.core.ops.gradients_impl import gradients


class Optimizer(object):
    def __init__(self, use_locking, name):
        if not name:
            raise ValueError('Must specify the optimizer name.')
        self._use_locking = use_locking
        self._name = name
        # Store the losses from gradients.
        self._targets = []
        # Store the external global step.
        self._global_step = None
        # Store the internal param updater.
        self.updater = None

    def apply_gradients(self, grads_and_vars, global_step=None):
        self._global_step = global_step
        grads_and_vars = list(grads_and_vars)

        # Firstly, we should extract the potential decays.
        l2_decays = []
        for grad, var in grads_and_vars:
            if hasattr(var, '__regularizer__'):
                if var .__regularizer__ and \
                        var.__regularizer__.l2 > 0:
                    l2_decays.append(var.__regularizer__.l2)

        # Find the base decay factor.
        self.updater.l2_decay = \
            base_l2_decay = min(l2_decays) \
            if len(l2_decays) > 0 else -1.

        # Add to targets.
        targets = set()
        for grad, var in grads_and_vars:
            decay_multiplier = 0.
            if hasattr(var, '__regularizer__'):
                if var.__regularizer__ and \
                        var.__regularizer__.l2 > 0:
                    decay_multiplier = \
                        var.__regularizer__.l2 / base_l2_decay
            self.updater.append((var, grad), decay_mult=decay_multiplier)
            if var._grad_info is not None:
                targets.update(var._grad_info.cost)
        self._targets.extend(list(targets))
        return self

    @classmethod
    def compute_gradients(cls, loss, var_list=None):
        if var_list is None:
            var_list = variables.trainable_variables() + \
                ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)
        grads = gradients(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))
        return grads_and_vars

    def get_name(self):
        return self._name

    def minimize(self, loss, global_step=None, var_list=None):
        self._global_step = global_step
        grads_and_vars = self.compute_gradients(loss, var_list)
        return self.apply_gradients(grads_and_vars, global_step)

    def _inc_global_step(self):
        """Increase the internal global step."""
        if self._global_step is not None:
            gs = int(self._global_step)
            if types.is_tensor(self._global_step):
                workspace.feed_tensor(
                    self._global_step, gs + 1,
                    enforce_cpu=True,
                )
            else:
                self._global_step += 1

    def _set_updater(self, cls, learning_rate, *args, **kwargs):
        """Set the updater and learning rate."""
        if types.is_tensor(learning_rate):
            base_lr = float(learning_rate)
            self.updater = cls(base_lr, *args, **kwargs)
            slot_lr = self.updater._slot + '/base_lr'
            workspace.set_tensor_alias(learning_rate, slot_lr)
            if types.is_symbolic_tensor(learning_rate):
                self._targets.append(learning_rate)
        else:
            self.updater = cls(learning_rate, *args, **kwargs)


class GradientDescentOptimizer(Optimizer):
    def __init__(
        self,
        learning_rate,
        use_locking=False,
        name='GradientDescent',
    ):
        super(GradientDescentOptimizer, self).__init__(use_locking, name)
        self._set_updater(updaters.SGD, learning_rate, momentum=0.,)


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
        cls = updaters.Nesterov if use_nesterov else updaters.SGD
        self._set_updater(cls, learning_rate, momentum=momentum)


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
        self._set_updater(
            updaters.Adam,
            learning_rate,
            beta1=beta1,
            beta2=beta2,
            eps=epsilon,
        )


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
        self._set_updater(
            updaters.RMSProp,
            learning_rate,
            momentum=momentum,
            decay=decay,
            eps=epsilon,
        )
