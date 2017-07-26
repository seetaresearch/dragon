# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from dragon.core.tensor import Tensor
import dragon.core.workspace as ws
import os

class ScalarSummary(object):
    def __init__(self, log_dir='logs'):
        self.log_dir = os.path.join(log_dir, 'scalar')
        if not os.path.exists(self.log_dir): os.makedirs(self.log_dir)

    def add_summary(self, scalar, global_step):
        if isinstance(scalar, Tensor):
            key, value = scalar.name, ws.FetchTensor(scalar)[0]
        elif isinstance(scalar, tuple): key, value = scalar
        else: raise TypeError()
        key = key.replace('/', '_')

        with open(os.path.join(self.log_dir, key + '.txt'), 'a') as f:
            f.write(str(global_step) + ' ' + str(value) + '\n')