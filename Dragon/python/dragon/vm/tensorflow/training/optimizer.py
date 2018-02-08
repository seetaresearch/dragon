# --------------------------------------------------------
# TensorFlow @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.updaters as updaters
import dragon.vm.theano as theano
import dragon.vm.theano.tensor as T

from dragon.vm.tensorflow.framework import ops
from dragon.vm.tensorflow.ops import variables


class Optimizer(object):
    def __init__(self, use_locking, name):
        if not name:
            raise ValueError('Must specify the optimizer name.')
        self._use_locking = use_locking
        self._name = name
        self._slots = {}
        self.loss = self.updater = None
        self.train = self.update = None

    def get_name(self):
        return self._name

    def minimize(self, loss, global_step=None, var_list=None, **kwargs):
        grads_and_vars = self.compute_gradients(loss, var_list)
        return self.apply_gradients(grads_and_vars, global_step=global_step)

    def compute_gradients(self, loss, var_list=None, **kwargs):
        if var_list is None:
            var_list = variables.trainable_variables() + \
                ops.get_collection(ops.GraphKeys.TRAINABLE_RESOURCE_VARIABLES)

        self.loss = loss
        grads = T.grad(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))
        return grads_and_vars

    def apply_gradients(self, grads_and_vars, global_step=None, **kwargs):
        objs = set()
        for grad_var in grads_and_vars:
            self.updater.append((grad_var[1], grad_var[0])) # (var, grad)
            for obj in grad_var[1].grad_objs: objs.add(obj)
        self.objs = list(objs)
        return self

    def run(self, feed_dict=None):
        # objective function
        if not hasattr(self, '_objective_func'):
            # find minimum solving targets
            targets = set()
            for t in self.objs: targets.add(t)
            if feed_dict is not None:
                self._objective_func = theano.function(inputs=feed_dict.keys(),
                                                       outputs=list(targets))
            else:
                self._objective_func = theano.function(outputs=list(targets))
        if feed_dict is not None:
            self._objective_func(*feed_dict.values())
        else:
            self._objective_func()

        # update function
        if not hasattr(self, '_update_func'):
            self._update_func = theano.function(updater=self.updater)
        self._update_func()


class GradientDescentOptimizer(Optimizer):
    def __init__(self, learning_rate, use_locking=False, name='GradientDescent'):
        super(GradientDescentOptimizer, self).__init__(use_locking, name)
        self.updater = updaters.SGDUpdater(learning_rate, 0.0)


class MomentumOptimizer(Optimizer):
    def __init__(self, learning_rate, momentum,
                 use_locking=False, name='Momentum', use_nesterov=False):
        super(MomentumOptimizer, self).__init__(use_locking, name)
        if not use_nesterov:
            self.updater = updaters.SGDUpdater(learning_rate, momentum)
        else:
            self.updater = updaters.NesterovUpdater(learning_rate, momentum)


class AdamOptimizer(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8,
                 use_locking=False, name='Adam'):
        super(AdamOptimizer, self).__init__(use_locking, name)
        self.updater = updaters.AdamUpdater(learning_rate, beta1, beta2, epsilon)


class RMSPropOptimizer(Optimizer):
    def __init__(self, learning_rate, decay, momentum, epsilon=1e-10,
                 use_locking=False, centered=False, name='RMSProp'):
        super(RMSPropOptimizer, self).__init__(use_locking, name)
        self.updater = updaters.RMSPropUpdater(learning_rate, decay, epsilon)
