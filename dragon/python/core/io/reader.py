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

import numpy
import multiprocessing

from dragon.core.framework import config


class DataReader(multiprocessing.Process):
    """Read examples from a dataset.

    The dataset class and data source are required to create a reader:

    ```python
    # Here we use ``dragon.io.KPLRecordDataset``
    dataset = dragon.io.KPLRecordDataset
    simple_reader = DataReader(dataset=dataset, source=path)
    ```

    Partition are available over distributed nodes:

    ```python
    distributed_reader = DataReader(
        dataset=dataset,
        source=path,
        part_idx=rank,
        num_parts=num_ranks,
    )
    ```

    There are two shuffle schemes:

    ```python
    # Recommendation: SSD or dataset is tiny
    example_wise_shuffle_reader = DataReader(
        dataset=dataset,
        source=path,
        shuffle=True,
        num_chunks=0,  # Set to the number of examples
    )

    # Recommendation: HDD or dataset is huge
    chunk_wise_shuffle_reader = DataReader(
        dataset=dataset,
        source=path,
        shuffle=True,
        num_chunks=2048,
    )
    ```

    """

    def __init__(self, **kwargs):
        """Create a ``DataReader``.

        Parameters
        ----------
        dataset : class
            The dataset class to load examples.
        source : str
            The path of data source.
        shuffle : bool, optional, default=False
            Whether to shuffle the data.r
        num_chunks : int, optional, default=0
            The number of chunks to split.
        num_parts : int, optional, default=1
            The number of partitions over dataset.
        part_idx : int, optional, default=0
            The index of current partition.
        seed : int, optional
            The random seed to use instead.

        """
        super(DataReader, self).__init__()
        self._dataset = kwargs.get('dataset', None)
        self._source = kwargs.get('source', '')
        self._shuffle = kwargs.get('shuffle', False)
        self._num_chunks = kwargs.get('num_chunks', 0)
        self._num_parts = kwargs.get('num_parts', 1)
        self._part_idx = kwargs.get('part_idx', 0)
        self._seed = kwargs.get('seed', config.config().random_seed)
        self._begin, self._end = 0, 0
        self._perm_size, self._perm = 1, None
        self._chunk_size, self._num_examples = 1, 1
        self._example_cursor, self._chunk_cursor = 0, 0
        self.q_out = None
        self.daemon = True

    def before_first(self):
        """Move the cursor before begin."""
        self._example_cursor = self._begin
        self._dataset.redirect(self._begin)

    def next_example(self):
        """Return the next example."""
        self._example_cursor += 1
        return self._dataset.get()

    def next_chunk(self):
        """Select the next chunk."""
        self._chunk_cursor += 1
        if self._chunk_cursor >= self._perm_size:
            self.reset()
        else:
            chunk_idx = self._part_idx * self._perm_size + int(self._perm[self._chunk_cursor])
            self._begin = chunk_idx * self._chunk_size
            if self._begin >= self._num_examples:
                self.next_chunk()
            else:
                self._end = min(self._begin + self._chunk_size, self._num_examples)
            self.before_first()

    def reset(self):
        """Reset the environment of dataset."""
        if self._num_parts > 1 or self._shuffle:
            self._chunk_cursor = -1
            self._part_idx = (self._part_idx + 1) % self._num_parts
            if self._shuffle:
                self._perm = numpy.random.permutation(self._perm_size)
            self.next_chunk()
        else:
            self._begin, self._end = 0, self._num_examples
            self.before_first()

    def run(self):
        """Start the process."""
        numpy.random.seed(self._seed)

        # Instantiate the dataset here to avoid a fork of process.
        # Fork will somehow fail if dataset is implemented in C/C++.
        self._dataset = self._dataset(self._source)
        self._num_examples = self._dataset.size

        # Determine the chunk scheme on different settings.
        def div_up(a, b):
            return (a + b - 1) // b

        if self._shuffle:
            if self._num_chunks <= 0:
                # Each chunk has at most 1 example (ExampleWise).
                self._perm_size = div_up(self._num_examples, self._num_parts)
            else:
                # Each chunk has several examples (ChunkWise).
                self._perm_size = div_up(self._num_chunks, self._num_parts)
                self._chunk_size = div_up(self._num_examples, self._num_chunks)
        else:
            # Each chunk has the examples of whole shard (ShardWise).
            self._chunk_size = div_up(self._num_examples, self._num_parts)

        # Reset the layout of permutation.
        self._perm = numpy.arange(self._perm_size)
        self.reset()

        # Persist a loop to read examples.
        while True:
            self.q_out.put(self.next_example())
            if self._example_cursor >= self._end:
                if self._num_parts > 1 or self._shuffle:
                    self.next_chunk()
                else:
                    self.reset()
