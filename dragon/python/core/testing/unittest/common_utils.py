# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Common unittest utilities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys
import unittest

import argparse
import dragon

# The global argument parser
parser = argparse.ArgumentParser(add_help=False)
build_info = dragon.sysconfig.get_build_info()

# The optional testing flags
TEST_CUDA = dragon.cuda.is_available()
TEST_MPS = dragon.mps.is_available()
TEST_MPI = dragon.distributed.is_mpi_available()
TEST_CUDNN_CONV3D_NHWC = build_info.get('cudnn_version', '0.0.0') > '8.0.0'


def run_tests(argv=None):
    """Run tests under the current ``__main__``."""
    if argv is None:
        args, remaining = parser.parse_known_args()
        argv = [sys.argv[0]] + remaining
    unittest.main(argv=argv)
