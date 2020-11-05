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
"""Utilities for TFRecord."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import numpy
import os
import struct
import zlib

from dragon.core.proto import tf_example_pb2


class TFRecordExample(object):
    """Describe an example of the TFRecord."""

    def __init__(self):
        """Create a ``TFRecordExample``."""
        self._proto = None
        self._features = collections.OrderedDict()

    @property
    def proto(self):
        """Pack the stored features into a protocol message."""
        if self._proto is None:
            features = tf_example_pb2.Features(feature=self._features)
            self._proto = tf_example_pb2.Example(features=features)
        return self._proto

    def add_floats(self, key, value):
        """Add a named float feature.

        Parameters
        ----------
        key : str
            The unique key.
        value : Sequence[float]
            A sequence of floats.

        """
        self._features[key] = tf_example_pb2.Feature(
            float_list=tf_example_pb2.FloatList(value=value),
        )

    def add_ints(self, key, value):
        """Add a named integer feature.

        Parameters
        ----------
        key : str
            The unique key.
        value : Sequence[int]
            A sequence of integers.

        """
        self._features[key] = tf_example_pb2.Feature(
            int64_list=tf_example_pb2.Int64List(value=value),
        )

    def add_strings(self, key, value):
        """Add a named string feature.

        Parameters
        ----------
        key : str
            The unique key.
        value : Sequence[bytes]
            A sequence of strings.

        """
        self._features[key] = tf_example_pb2.Feature(
            bytes_list=tf_example_pb2.BytesList(value=value),
        )

    def reset(self):
        """Reset the message."""
        self._proto = None
        self._features = collections.OrderedDict()

    def serialize_to(self, pack_length=True, pack_crc32=True):
        """Serialize the message to raw bytes.

        Parameters
        ----------
        pack_length : bool, optional, default=True
            **True** to pack with length.
        pack_crc32 : bool, optional, default=True
            **True** to pack with crc32.

        Returns
        -------
        bytes
            The serialized message bytes.

        """

        def mask_crc32(value):
            """
            Calculate a 32 - bit 32 - bit integer.

            Args:
                value: (array): write your description
            """
            crc = zlib.crc32(bytes(value))
            crc = crc & 0xffffffff if crc < 0 else crc
            crc = numpy.array(crc, 'uint32')
            crc = (crc >> 15) | (crc << 17).astype('uint32')
            return int((crc + 0xa282ead8).astype('uint32'))

        bytes_seq = []
        proto_bytes = self.proto.SerializeToString()

        if pack_length:
            length = len(proto_bytes)
            bytes_seq.append(struct.pack('q', length))
            if pack_crc32:
                length_crc = mask_crc32(length)
                bytes_seq.append(struct.pack('I', length_crc))

        bytes_seq.append(proto_bytes)
        if pack_crc32:
            proto_crc = mask_crc32(proto_bytes)
            bytes_seq.append(struct.pack('I', proto_crc))

        if len(bytes_seq) == 1:
            return bytes_seq[0]

        bytes_all = bytes()
        for b in bytes_seq:
            bytes_all += b
        return bytes_all


class TFRecordWriter(object):
    """Write examples into the TFRecord.

    To write the ``TFRecord``, a descriptor is required
    to describe the features for different examples:

    ```python
    features = "{
        'data': tf.FixedLenFeature([], tf.string, ''),
        'label': tf.VarLenFeature(tf.int64, -1),
    }"
    ```

    Then you should fill a example with features:

    ```python
    example = dragon.io.TFRecordExample()
    example.add_strings('data', [img_bytes])
    example.add_ints('shape', img_shape)
    example.add_ints('label', labels)
    ```

    Finally, open a writer to write this example:

    ```python
    with dragon.io.TFRecordWriter(path, features) as writer:
        writer.write(example)
    ```

    For how to fill examples, see ``dragon.io.TFRecordExample``.

    """

    def __init__(
        self,
        path,
        features,
        max_examples=2**63 - 1,
        zfill_width=5,
    ):
        """Create a ``TFRecordWriter``.

        Parameters
        ----------
        path : str
            The path to write the record file.
        features : str
            The descriptor for reading.
        max_examples : int, optional
            The max examples of a single record file.
        zfill_width : int, optional, default=5
            The width of zfill for naming record files.

        """
        self._writing = True
        self._shard_id, self._examples = -1, 0
        self._max_examples = max_examples
        self._data_template = path + '/{0:0%d}.data' % zfill_width
        self._index_template = path + '/{0:0%d}.index' % zfill_width
        features_file = path + '/FEATURES'
        if not os.path.exists(path):
            os.makedirs(path)
        self._data_writer = None
        self._index_writer = None
        with open(features_file, 'w') as f:
            f.write(features)
        self._maybe_new_shard()

    def write(self, example):
        """Write a example to the file.

        Parameters
        ----------
        example : dragon.io.TFRecordExample
            The data example.

        """
        if self._writing:
            current = self._data_writer.tell()
            self._data_writer.write(example.serialize_to())
            self._index_writer.write(
                str(current) + ' ' +
                str(self._data_writer.tell() - current) + '\n')
            self._examples += 1
            self._maybe_new_shard()
        else:
            raise RuntimeError('Writer has been closed.')

    def close(self):
        """Close the file."""
        if self._writing:
            self._data_writer.close()
            self._index_writer.close()
            self._writing = False

    def _maybe_new_shard(self):
        """
        Create shard shard shard file.

        Args:
            self: (todo): write your description
        """
        if self._examples >= self._max_examples or \
                self._data_writer is None:
            self._examples = 0
            self._shard_id += 1
            data_file = self._data_template.format(self._shard_id)
            index_file = self._index_template.format(self._shard_id)
            for file in (data_file, index_file):
                if os.path.exists(file):
                    raise ValueError('File %s existed.' % file)
            if self._data_writer is not None:
                self._data_writer.close()
                self._index_writer.close()
            self._data_writer = open(data_file, 'wb')
            self._index_writer = open(index_file, 'w')

    def __enter__(self):
        """Enter a **with** block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a **with** block and close the file."""
        self.close()

    def __del__(self):
        """Delete writer and close the file."""
        self.close()
