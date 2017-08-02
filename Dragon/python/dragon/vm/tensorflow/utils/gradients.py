# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import dragon.vm.theano.tensor as T

__all__ = ['gradients']

def gradients(loss, var_list=None):
    if var_list is None:
        from dragon.vm.tensorflow.core.variables import TRAINABLE_VARIABLES
        global TRAINABLE_VARIABLES
        var_list = TRAINABLE_VARIABLES.values()
    grads = T.grad(loss, var_list)
    grads_and_vars = list(zip(grads, var_list))
    return grads_and_vars