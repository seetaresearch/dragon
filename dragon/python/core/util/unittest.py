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

import sys
import unittest

import argparse
import dragon

from dragon.vm import torch as torch_vm


parser = argparse.ArgumentParser(add_help=False)
TEST_CUDA = dragon.cuda.is_available()


def new_tensor(data, constructor='EagerTensor', execution=None):
    if execution is not None:
        if execution == 'GRAPH_MODE':
            return dragon.Tensor(
                shape=data.shape,
                dtype=str(data.dtype),
            ).set_value(data)
        else:
            return dragon.EagerTensor(data, copy=True)
    if constructor == 'EagerTensor':
        return dragon.EagerTensor(data, copy=True)
    elif constructor == 'Tensor':
        return dragon.Tensor(
            shape=data.shape,
            dtype=str(data.dtype),
        ).set_value(data)
    elif constructor == 'torch.Tensor':
        return torch_vm.tensor(data)
    else:
        raise ValueError('Unknown constructor:', constructor)


def run_tests(argv=None):
    """Run tests under the current ``__main__``."""
    if argv is None:
        args, remaining = parser.parse_known_args()
        argv = [sys.argv[0]] + remaining
    unittest.main(argv=argv)
