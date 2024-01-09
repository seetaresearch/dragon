# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Common unittest utilities."""

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
TEST_MLU = dragon.mlu.is_available()
TEST_MPI = dragon.distributed.is_mpi_available()
TEST_CUDNN_CONV3D_NHWC = build_info.get("cudnn_version", "0.0.0") > "8.0.0"


def run_tests(argv=None):
    """Run tests under the current ``__main__``."""
    if argv is None:
        args, remaining = parser.parse_known_args()
        argv = [sys.argv[0]] + remaining
    unittest.main(argv=argv)
