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

import collections
import multiprocessing as mp
import os

try:
    from nvidia.dali import ops
    from nvidia.dali import tfrecord
except ImportError:
    from dragon.core.util import deprecation
    ops = deprecation.NotInstalled('nvidia.dali')
    tfrecord = deprecation.NotInstalled('nvidia.dali')
try:
    import codewithgpu
except ImportError:
    codewithgpu = deprecation.NotInstalled('codewithgpu')

from dragon.vm.dali.core.framework import context
from dragon.vm.dali.core.ops.builtin_ops import ExternalSource


class CGRecordReader(object):
    """Read examples from the CGRecord.

    Examples:

    ```python
    class MyPipeline(dali.Pipeline):

        def __init__():
            super(MyPipeline, self).__init__()
            # Assume that we have the following files:
            # /path/to/records/00000.data
            # /path/to/records/00000.index
            # /path/to/records/METADATA
            self.reader = dali.ops.CGRecordReader(
                path='/path/to/records'
                features=('image', 'label'),
                pipeline=self,
                # Shuffle locally in the next ``initial_fill`` examples
                # It turns to be weak with the decreasing of ``initial_fill``
                # and disabled if ``initial_fill`` is set to **1**
                random_shuffle=True, initial_fill=1024)

        def iter_step(self):
            self.reader.feed_inputs()

        def define_graph(self):
            inputs = self.reader()
    ```

    """

    def __init__(
        self,
        path,
        features,
        pipeline,
        shard_id=0,
        num_shards=1,
        random_shuffle=False,
        initial_fill=1024,
        **kwargs
    ):
        """Create a ``KPLRecordReader``.

        Parameters
        ----------
        path : str
            The folder of record files.
        features : Sequence[str], required
            The name of features to extract.
        pipeline : nvidia.dali.Pipeline, required
            The pipeline to connect to.
        shard_id : int, optional, default=0
            The index of partition to read.
        num_shards : int, optional, default=1
            The total number of partitions over dataset.
        random_shuffle : bool, optional, default=False
            Whether to shuffle the data.
        initial_fill : int, optional, default=1024
            The length of sampling sequence for shuffle.

        """
        self._pipe = pipeline
        self._batch_size = pipeline.batch_size
        self._prefetch_depth = pipeline._prefetch_queue_depth
        self._buffer = mp.Queue(self._prefetch_depth * self._batch_size)
        self._dataset_reader = codewithgpu.DatasetReader(
            path=path, output_queue=self._buffer,
            partition_idx=shard_id, num_partitions=num_shards,
            shuffle=random_shuffle, initial_fill=initial_fill, **kwargs)
        self._dataset_reader.start()
        with context.device('cpu'):
            self.features = dict((k, ExternalSource()) for k in features)

        def cleanup():
            self.terminate()

        import atexit
        atexit.register(cleanup)

    def example_to_data(self, example):
        """Define the translation from example to array data.

        Override this method to implement the translation.

        """
        raise NotImplementedError

    def feed_inputs(self):
        """Feed the data to edge references.

        Call this method in the ``Pipeline.iter_setup(...)``.

        """
        feed_dict = collections.defaultdict(list)
        for i in range(self._pipe.batch_size):
            data = self.example_to_data(self._buffer.get())
            for k, v in data.items():
                feed_dict[k].append(v)
        for k, v in self.features.items():
            self._pipe.feed_input(self.features[k], feed_dict[k])

    def terminate(self):
        """Terminate the reader."""
        self._dataset_reader.terminate()
        self._dataset_reader.join()

    def __call__(self, *args, **kwargs):
        """Create the edge references for features.

        Call this method in the ``Pipeline.define_graph(...)``.

        Returns
        -------
        Dict[str, _EdgeReference]
            The feature reference dict.

        """
        self.features = dict((k, v()) for k, v in self.features.items())
        return self.features


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
            features = f.read()
            features = features.replace('tf.', 'tfrecord.')
            features = features.replace('tf.io.', 'tfrecord.')
            features = eval(features)
        data_files.sort()
        index_files.sort()
        data = [os.path.join(path, e) for e in data_files]
        index = [os.path.join(path, e) for e in index_files]
        return data, index, features
