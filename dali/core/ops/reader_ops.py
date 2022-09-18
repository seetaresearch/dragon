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
"""Reader ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import json
import os

try:
    from nvidia.dali import ops
    from nvidia.dali import tfrecord as tfrec
except ImportError:
    from dragon.core.util import deprecation
    ops = deprecation.NotInstalled('nvidia.dali')
    tfrec = deprecation.NotInstalled('nvidia.dali')


class TFRecordReader(object):
    """Read examples from the TFRecord.

    Examples:

    ```python
    # Assume that we have the following files:
    # /path/to/records/00000.data
    # /path/to/records/00000.index
    # /path/to/records/METADATA
    input = dali.ops.TFRecordReader(
        path='/path/to/records',
        # Shuffle locally in the next ``initial_fill`` examples
        # It turns to be weak with the decreasing of ``initial_fill``
        # and disabled if ``initial_fill`` is set to **1**
        random_shuffle=True, initial_fill=1024)
    ```

    """

    def __new__(
        cls,
        path,
        shard_id=0,
        num_shards=1,
        random_shuffle=False,
        initial_fill=1024,
        **kwargs
    ):
        """Create a ``TFRecordReader``.

        Parameters
        ----------
        path : str
            The folder of record files.
        shard_id : int, optional, default=0
            The index of partition to read.
        num_shards : int, optional, default=1
            The total number of partitions over dataset.
        random_shuffle : bool, optional, default=False
            Whether to shuffle the data.
        initial_fill : int, optional, default=1024
            The length of sampling sequence for shuffle.

        Returns
        -------
        nvidia.dali.ops.readers.TFRecord
            The reader instance.

        """
        path, index_path, features = cls.check_files(path)
        return ops.readers.TFRecord(
            path=path,
            index_path=index_path,
            shard_id=shard_id,
            num_shards=num_shards,
            features=features,
            random_shuffle=random_shuffle,
            initial_fill=initial_fill,
            **kwargs
        )

    @staticmethod
    def check_files(path):
        data_files, index_files, meta_data_file = [], [], None
        for file in os.listdir(path):
            if file.endswith('.data'):
                data_files.append(file)
            elif file.endswith('.index'):
                index_files.append(file)
            elif file == 'METADATA':
                meta_data_file = file
        if meta_data_file is None:
            raise FileNotFoundError('Excepted meta data file: %s' % meta_data_file)
        with open(os.path.join(path, meta_data_file), 'r') as f:
            features = json.load(f)['features']
            for k in list(features.keys()):
                shape, dtype, default_value = features[k]
                dtype = getattr(tfrec, 'string' if dtype == 'bytes' else dtype)
                if shape is None:
                    features[k] = tfrec.VarLenFeature(dtype, default_value)
                else:
                    features[k] = tfrec.FixedLenFeature(shape, dtype, default_value)
        data_files.sort()
        index_files.sort()
        data = [os.path.join(path, e) for e in data_files]
        index = [os.path.join(path, e) for e in index_files]
        return data, index, features
