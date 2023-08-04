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
"""Test distributed module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import unittest

import dragon
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.testing.unittest.common_utils import TEST_MPI


class TestBackend(unittest.TestCase):
    """Test backend components."""

    def test_empty_group(self):
        for backend in (None, 'AUTO', 'NCCL', 'CNCL', 'MPI', 'UNKNOWN', 0):
            try:
                group = dragon.distributed.new_group(backend=backend)
                self.assertEqual(group.ranks, None)
                self.assertEqual(group.size, 0)
                self.assertEqual(group.arguments['backend'], group.backend)
                self.assertEqual(repr(group), '%s:None' % group.backend)
                with group.as_default():
                    self.assertEqual(dragon.distributed.get_group(), None)
                    self.assertEqual(dragon.distributed.get_backend(group), group.backend)
            except ValueError:
                pass

    @unittest.skipIf(not TEST_MPI, 'MPI unavailable')
    def test_mpi_single_process(self):
        self.assertEqual(dragon.distributed.get_rank(), 0)
        self.assertEqual(dragon.distributed.get_world_size(), 1)
        group = dragon.distributed.new_group(ranks=[0], backend='MPI')
        with group.as_default():
            self.assertEqual(dragon.distributed.get_rank(group), 0)

    @unittest.skipIf(not TEST_MPI, 'MPI unavailable')
    def test_finalize(self):
        dragon.distributed.finalize()


if __name__ == '__main__':
    run_tests()
