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

    class PartBoundaries(object):
        """Record the boundary of current part."""

        def __init__(self, start, end):
            self.start, self.end = start, end

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
        super(DataReader, self).__init__()
        self._dataset = kwargs.get('dataset', None)
        self._source = kwargs.get('source', '')
        self._part_idx = kwargs.get('part_idx', 0)
        self._num_parts = kwargs.get('num_parts', 1)
        self._shuffle = kwargs.get('shuffle', False)
        self._initial_fill = kwargs.get('initial_fill', 1024) if self._shuffle else 1
        self._seed = kwargs.get('seed', config.config().random_seed)
        self._first, self._cursor, self._last = 0, 0, 0
        self._part_size = 0
        self._num_examples = 0
        self._example_buffer = []
        self._parts = []
        self.q_out = None
        self.daemon = True

    def before_first(self):
        """Move the cursor before begin."""
        self._cursor = self._first
        self._dataset.redirect(self._first)

    def next_example(self):
        """Return the next example."""
        self._cursor += 1
        return self._dataset.get()

    def reset(self, stick_to_part=False):
        """Reset the environment of dataset."""
        # Redirect to the adjacent part if available.
        if not stick_to_part:
            self._part_idx = (self._part_idx + 1) % self._num_parts
        self._first = self._part_idx * self._part_size
        self._last = min(self._first + self._part_size, self._num_examples)
        self.before_first()
        # Use the new boundaries to avoid sampling duplicates
        # when buffer size is greater than dataset size.
        counter = self._parts[-1].end
        self._parts.append(DataReader.PartBoundaries(counter, counter))

    def run(self):
        """Start the process."""
        self._init_dataset()
        # Persist a loop to read examples.
        while True:
            # Pop the depleted part if necessary
            if self._parts[0].start == self._parts[0].end:
                self._parts.pop(0)
            offset = 0
            if self._shuffle:
                # Sample a random offset if shuffle required.
                offset = self._parts[0].end - self._parts[0].start
                offset = int(numpy.random.uniform(high=offset))
            # Choose a loaded example from the buffer.
            i = self._parts[0].start % len(self._example_buffer)
            j = (self._parts[0].start + offset) % len(self._example_buffer)
            self.q_out.put(self._example_buffer[j])
            self._example_buffer[j] = self._example_buffer[i]
            # Load and push back a new example into the buffer.
            k = self._parts[-1].end % len(self._example_buffer)
            self._example_buffer[k] = self.next_example()
            # Increase the part boundaries
            self._parts[-1].end += 1
            self._parts[0].start += 1
            # Reset the cursor if necessary
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
        self._parts.append(DataReader.PartBoundaries(0, 0))

        # Fill the initial buffer to support random sampling.
        self.reset(stick_to_part=True)
        for i in range(self._initial_fill):
            self._example_buffer.append(self.next_example())
            self._parts[-1].end += 1
            if self._cursor >= self._last:
                self.reset()
