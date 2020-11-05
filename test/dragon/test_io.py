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
"""Test the io module."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import shutil
import unittest

import dragon
from dragon.core.io.kpl_record import KPLRecordProtocol
from dragon.core.testing.unittest.common_utils import run_tests
from dragon.core.util import serialization


class TestKPLRecord(unittest.TestCase):
    """Test the kpl record components."""

    def test_writer_and_reader(self):
        """
        Serialize the writer and write it to disk.

        Args:
            self: (todo): write your description
        """
        path = '/tmp/test_dragon_io_kpl_record'
        protocol = {'a': ['float64'], 'b': ['int64'], 'c': ['bytes']}
        example = {'a': [1., 2., 3.], 'b': [4, 5, 6], 'c': [b'7', b'8', b'9']}
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            os.makedirs(path)
            with dragon.io.KPLRecordWriter(path, protocol) as writer:
                writer.write(example)
            try:
                writer.write(example)
            except RuntimeError:
                pass
            dataset = dragon.io.KPLRecordDataset(path=path)
            self.assertEqual(dataset.protocol, KPLRecordProtocol(protocol))
            self.assertEqual(dataset.size, 1)
            self.assertEqual(len(dataset), 1)
            dataset.redirect(0)
            self.assertEqual(example, dataset.get())
            for shuffle, initial_fill in [(False, 1), (True, 1), (True, 1024)]:
                reader = dragon.io.DataReader(
                    dataset=dragon.io.KPLRecordDataset,
                    source=path, shuffle=shuffle, initial_fill=initial_fill)
                reader._init_dataset()
        except (OSError, PermissionError):
            pass


class TestTFRecord(unittest.TestCase):
    """Test the tf record components."""

    def test_example(self):
        """
        Reads an example.

        Args:
            self: (todo): write your description
        """
        example = dragon.io.TFRecordExample()
        example.add_floats('a', [1., 2., 3.])
        example.add_ints('b', [4, 5, 6])
        example.add_strings('c', [b'7', b'8', b'9'])
        s = example.serialize_to(pack_length=False, pack_crc32=False)
        self.assertEqual(s, example.serialize_to()[12:-4])
        self.assertEqual(s, example.serialize_to(pack_length=False)[:-4])
        self.assertEqual(s, example.serialize_to(pack_crc32=False)[8:])
        proto1 = example.proto
        example.reset()
        proto2 = example.proto
        self.assertNotEqual(proto1, proto2)
        self.assertEqual(proto1, serialization.deserialize_proto(s, proto2))

    def test_writer(self):
        """
        Writes a single writer exists.

        Args:
            self: (todo): write your description
        """
        path = '/tmp/test_dragon_io_tf_record'
        try:
            if os.path.exists(path):
                shutil.rmtree(path)
            with dragon.io.TFRecordWriter(
                    path, features='', max_examples=1) as writer:
                writer.write(dragon.io.TFRecordExample())
                writer.write(dragon.io.TFRecordExample())
            try:
                writer.write(dragon.io.TFRecordExample())
            except RuntimeError:
                pass
        except (OSError, PermissionError):
            pass


if __name__ == '__main__':
    run_tests()
