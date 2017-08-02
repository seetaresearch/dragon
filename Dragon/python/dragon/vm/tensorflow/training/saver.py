# --------------------------------------------------------
# TensorFlow for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

__all__ = ['Saver']

import dragon.core.workspace as ws
from dragon.core.tensor import Tensor

class Saver(object):
    def __init__(self,
                 var_list=None,
                 max_to_keep=5,
                 name=None,):
        self.var_list = var_list

    def save(self,
             sess,
             save_path,
             global_step=None):
        from ..core.variables import VARIABLES
        global VARIABLES
        var_list = VARIABLES if self.var_list is None else self.var_list
        filename = save_path
        if global_step is not None:
            if isinstance(global_step, Tensor):
                __ndarray__global_step = ws.FetchTensor(global_step)
                if __ndarray__global_step.size != 1:
                    raise ValueError('global step must be a scalar of length 1.')
                filename += '-' + str(__ndarray__global_step.flatten()[0])
        ws.Snapshot(var_list.values(), filename=filename, suffix='')

    def restore(self, sess, save_path):
        ws.Restore(save_path)