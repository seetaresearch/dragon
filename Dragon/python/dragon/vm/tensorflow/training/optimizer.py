# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.updaters as updaters
import dragon.vm.theano.tensor as T
import dragon.vm.theano as theano
from dragon.vm.tensorflow.utils.check import feed_check

__all__ = [
    'GradientDescentOptimizer',
    'MomentumOptimizer',
    'AdamOptimizer',
    'RMSPropOptimizer'
]

class BaseOptimizer(object):
    def __init__(self):
        super(BaseOptimizer, self).__init__()
        self.loss = self.updater = None
        self.train = self.update = None

    def minimize(self, loss, var_list=None):
        grads_and_vars = self.compute_gradients(loss, var_list)
        self.apply_gradients(grads_and_vars)
        return self

    def compute_gradients(self, loss, var_list=None):
        if var_list is None:
            from dragon.vm.tensorflow.core.variables import TRAINABLE_VARIABLES
            global TRAINABLE_VARIABLES
            var_list = TRAINABLE_VARIABLES.values()
        self.loss = loss
        grads = T.grad(loss, var_list)
        grads_and_vars = list(zip(grads, var_list))
        return grads_and_vars

    def apply_gradients(self, grads_and_vars):
        objs = set()
        for grad_var in grads_and_vars:
            self.updater.append((grad_var[1], grad_var[0]))
            for obj in grad_var[0].grad_objs: objs.add(obj)
        self.objs = list(objs)
        return self

    def run(self, feed_dict=None):
        # make training function
        if self.train is None:
            if feed_dict is not None:
                feed_check(feed_dict)
                self.train = theano.function(inputs=feed_dict.keys(), outputs=self.loss)
            else: self.train =  theano.function(outputs=self.loss)

        # make update
        if self.update is None:
            self.update = theano.function(updater=self.updater)

        # training - execute
        if feed_dict is not None:
            loss = self.train(*feed_dict.values())
        else: loss = self.train()

        # update - execute
        self.update()

        return loss

class GradientDescentOptimizer(BaseOptimizer):
    def __init__(self, learning_rate):
        super(GradientDescentOptimizer, self).__init__()
        self.updater = updaters.SGDUpdater(learning_rate, 0.0)

class MomentumOptimizer(BaseOptimizer):
    def __init__(self, learning_rate, momentum):
        super(MomentumOptimizer, self).__init__()
        self.updater = updaters.SGDUpdater(learning_rate, momentum)

class AdamOptimizer(BaseOptimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        super(AdamOptimizer, self).__init__()
        self.updater = updaters.AdamUpdater(learning_rate, beta1, beta2, epsilon)

class RMSPropOptimizer(BaseOptimizer):
    def __init__(self, learning_rate, decay, momentum, epsilon=1e-10):
        super(RMSPropOptimizer, self).__init__()
        self.updater = updaters.RMSPropUpdater(learning_rate, decay, epsilon)


