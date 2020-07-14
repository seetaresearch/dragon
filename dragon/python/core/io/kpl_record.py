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
"""Utilities for KPLRecord."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

try:
    from kpl_dataset.dataset import ReadOnlyDataset
    from kpl_dataset.dataset import WriteOnlyDataset
    from kpl_dataset.record import BasicType
    from kpl_dataset.record import RecordDefine

    class KPLRecordMessage(object):
        """An enum-like class of available messages."""

        BYTES = BasicType.ByteArray
        BYTEARRAY = BasicType.ByteArray
        FLOAT = BasicType.Float
        FLOAT64 = BasicType.Float
        INT = BasicType.Int
        INT64 = BasicType.Int
        STRING = BasicType.String

except ImportError:
    from dragon.core.util import deprecation
    BasicType = deprecation.not_installed('kpl-dataset')
    RecordDefine = deprecation.not_installed('kpl-dataset')
    ReadOnlyDataset = deprecation.not_installed('kpl-dataset')
    WriteOnlyDataset = deprecation.not_installed('kpl-dataset')

from dragon.core.util import six


class KPLRecordProtocol(object):
    """Create a protocol for KPLRecord."""

    def __new__(cls, descriptor):
        return cls.canonicalize(descriptor)

    @classmethod
    def canonicalize(cls, descriptor):
        """Canonicalize the descriptor for protocol."""
        if isinstance(descriptor, dict):
            for k, v in descriptor.items():
                descriptor[k] = cls.canonicalize(v)
            return descriptor
        elif isinstance(descriptor, list):
            return [cls.canonicalize(v) for v in descriptor]
        else:
            return cls.get_message(descriptor)

    @classmethod
    def get_message(cls, descriptor):
        """Return the message from string descriptor."""
        if isinstance(descriptor, six.string_types):
            return getattr(KPLRecordMessage, descriptor.upper())
        return descriptor


class KPLRecordWriter(object):
    """Write examples into the KPLRecord.

    To write the ``KPLRecord``, a protocol is required
    to describe the structure for different examples:

    ```python
    # Nest descriptors can be used: (dict, list)
    # Type descriptors can be used: (bytes, float64, int64, string)
    my_protocol = {
        'data': 'bytes',
        'shape': ['int64'],
        'object': [{
            'name': 'string',
            'bbox': ['float64'],
        }]
    }
    ```

    Then, you can open a writer to write corresponding examples:

    ```python
    with dragon.io.KPLRecordWriter(path, my_protocol) as writer:
        writer.write({
            'data': image.tostring(),
            'shape': [300, 500, 3],
            'object': [{
                'name': 'cat',
                'bbox': [0., 0., 100., 200.],
            }
        })
    ```

    For how to read the record file, see ``dragon.io.KPLRecordDataset``.

    """

    def __init__(self, path, protocol, name=None):
        """Create a ``KPLRecordWriter``.

        Parameters
        ----------
        path : str
            The path to write the record file.
        protocol : Union[str, List, Dict]
            The descriptor of example structure.
        name : str, optional
            The optional dataset name.

        """
        self._writing = True
        protocol = KPLRecordProtocol(protocol)
        self._writer = WriteOnlyDataset(path, name, protocol)

    def write(self, example):
        """Write a example into the file.

        Parameters
        ----------
        example : Union[bytes, float, int, List, Dict]
            The data corresponding to protocol.

        """
        if self._writing:
            self._writer.write(example)
        else:
            raise RuntimeError('Writer has been closed.')

    def close(self):
        """Close the file."""
        if self._writing:
            self._writer.close()
            self._writing = False

    def __enter__(self):
        """Enter a **with** block."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Exit a **with** block and close the file."""
        self.close()

    def __del__(self):
        """Delete writer and close the file."""
        self.close()


class KPLRecordDataset(object):
    """Dataset to load the KPLRecord.

    You can create this dataset for ``DataReader``:

    ```python
    reader = dragon.io.DataReader(
        dataset=dragon.io.KPLRecordDataset,
        source=path,
    )
    ```

    For the detailed reading procedure, see ``DataReader``.

    """

    def __init__(self, path, name=None):
        """Create a ``KPLRecordDataset``.

        Parameters
        ----------
        path : str
            The path of record file.
        name : str, optional
            The optional dataset name.

        """
        self._env = ReadOnlyDataset(path, name)
        self._env._init_data_handle()
        self._env._data_open = True
        self.redirect(0)  # Seek first for partial indices.

    @property
    def protocol(self):
        """Return the protocol of dataset.

        Returns
        -------
        RecordType
            The descriptor of dataset.

        """
        return self._env.record_type

    @property
    def size(self):
        """Return the total number of examples.

        Returns
        -------
        int
            The number of examples.

        """
        return self._env._record_count

    def get(self):
        """Pop a example starting from cursor.

        Returns
        -------
        ProtocolType
            The example.

        """
        return self._env.next()

    def redirect(self, index):
        """Move the cursor to the specified index.

        Parameters
        ----------
        index : int
            The index to move.

        """
        self._env._cursor = index
        self._env._data_handle.seek(self._env._index[index])

    def __len__(self):
        """Return the total number of examples."""
        return self.size

    def __del__(self):
        """Close the environment"""
        self._env.close()
