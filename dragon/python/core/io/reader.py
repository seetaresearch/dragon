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
"""Process to read the distributed data."""

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

    Shuffle is supported to randomly sampling into a sequence buffer:

    ```python
    shuffle_reader = DataReader(
        dataset=dataset,
        source=path,
        shuffle=True,
        # It is recommended to set a buffer size larger than
        # the batch size to make batches of single node more diverse.
        # Default value 1024 is sufficient for most case.
        initial_fill=1024,
    )
    ```

    Partition are available over distributed nodes:

    ```python
    distributed_reader = DataReader(
        dataset=dataset,
        source=path,
        part_idx=rank,
        num_parts=world_size,
    )
    ```

    """

    class BufferBound(object):
        """Record the boundary of current buffer."""

        def __init__(self, start, end):
            self.start, self.end = start, end

        @property
        def is_depleted(self):
            return self.start == self.end

    def __init__(self, **kwargs):
        """Create a ``DataReader``.

        Parameters
        ----------
        dataset : class
            The dataset class to load examples.
        source : str
            The path of data source.
        part_idx : int, optional, default=0
            The index of partition to read.
        num_parts : int, optional, default=1
            The total number of partitions over dataset.
        shuffle : bool, optional, default=False
            Whether to shuffle the data.
        initial_fill : int, optional, default=1024
            The length of sampling sequence for shuffle.
        seed : int, optional
            The random seed to use instead.

        """
        super(DataReader, self).__init__(daemon=True)
        self._dataset = kwargs.get('dataset', None)
        self._source = kwargs.get('source', '')
        self._part_idx = kwargs.get('part_idx', 0)
        self._num_parts = kwargs.get('num_parts', 1)
        self._shuffle = kwargs.get('shuffle', False)
        self._initial_fill = kwargs.get('initial_fill', 1024)
        self._seed = kwargs.get('seed', config.config().random_seed)
        self._stick_to_part = kwargs.get('stick_to_part', True)
        self._first, self._cursor, self._last = 0, 0, 0
        self._part_size = 0
        self._num_examples = 0
        self._buffer_seq = []
        self._buffer_bounds = []
        self._reader_queue = None

    def before_first(self):
        """Move the cursor before begin."""
        self._cursor = self._first
        self._dataset.redirect(self._first)

    def next_example(self):
        """Return the next example."""
        self._cursor += 1
        return self._dataset.get()

    def reset(self):
        """Reset the environment of dataset."""
        # Redirect to the adjacent part if available.
        if not self._stick_to_part:
            self._part_idx = (self._part_idx + 1) % self._num_parts
        self._first = self._part_idx * self._part_size
        self._last = min(self._first + self._part_size, self._num_examples)
        self.before_first()
        # Use new boundary to avoid sampling duplicates
        # when buffer size is greater than dataset size.
        counter = self._buffer_bounds[-1].end
        self._buffer_bounds.append(self.BufferBound(counter, counter))

    def run(self):
        """Start the process."""
        self._init_dataset()
        # Persist a loop to read examples.
        while True:
            # Pop the depleted buffer if necessary.
            if self._buffer_bounds[0].is_depleted:
                self._buffer_bounds.pop(0)
            pop_bound = self._buffer_bounds[0]
            push_bound = self._buffer_bounds[-1]
            pop_offset = 0
            if self._shuffle:
                # Sample a random offset.
                pop_range = pop_bound.end - pop_bound.start
                pop_offset = numpy.random.randint(0, pop_range)
            # Pop an example from the buffer.
            i = pop_bound.start % len(self._buffer_seq)
            j = (pop_bound.start + pop_offset) % len(self._buffer_seq)
            self._reader_queue.put(self._buffer_seq[j])
            self._buffer_seq[j] = self._buffer_seq[i]
            # Push an example into the buffer.
            k = push_bound.end % len(self._buffer_seq)
            self._buffer_seq[k] = self.next_example()
            # Increase the buffer boundary.
            push_bound.end += 1
            pop_bound.start += 1
            # Reset the cursor if necessary.
            if self._cursor >= self._last:
                self.reset()

    def _init_dataset(self):
        """Initialize the dataset."""
        numpy.random.seed(self._seed)

        # Instantiate the dataset here to avoid a fork of process.
        # Fork will somehow fail if dataset is implemented in C/C++.
        self._dataset = self._dataset(self._source)

        # Determine the part specification.
        self._num_examples = self._dataset.size
        self._part_size = (self._num_examples + self._num_parts - 1) // self._num_parts

        # Fill the initial buffer to support random sampling.
        self._buffer_bounds.append(self.BufferBound(0, 0))
        self.reset()
        for _ in range(self._initial_fill):
            self._buffer_bounds[-1].end += 1
            self._buffer_seq.append(self.next_example())
            if self._cursor >= self._last:
                self.reset()
