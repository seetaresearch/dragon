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

import multiprocessing as mp
import queue
import threading
import time

import numpy

from dragon.core import distributed
from dragon.core.util import logging
from dragon.core.io import reader
from dragon.utils.vision import data_worker


class DataIterator(threading.Thread):
    """Iterator to return the batch of data for image classification.

    Usually, we will pack serialized data into ``KPLRecord``:

    ```python
    writer = dragon.io.KPLRecordWriter(
        path,
        protocol={
            'data': 'bytes',  # Content of image
            'encoded': 'int64',  # Image is encoded?
            'shape': ['int64'],  # (H, W, C)
            'label': ['int64'],  # Label index
        }
    )
    for example in examples:
        writer.write(example)
    ```

    Defining an iterator will start the prefetch processes:

    ```python
    iterator = dragon.vision.DataIterator(
        dataset=dragon.io.KPLRecordDataset,
        source=path,
        batch_size=32,
        shuffle=True,
        phase='TRAIN',  # Flag to determine some methods
    )
    ```

    Then, you can get a batch of data by ``Iterator.next()``:

    ```python
    images, labels = iterator.next()
    ```

    """

    def __init__(self, **kwargs):
        """Create a ``DataIterator``.

        Parameters
        ----------
        dataset : class
            The dataset class to load examples.
        source : str
            The path of data source.
        shuffle : bool, optional, default=False
            Whether to shuffle the data.
        initial_fill : int, optional, default=1024
            The length of sampling sequence for shuffle.
        resize : int, optional, default=0
            The size for the shortest edge.
        padding : int, optional, default=0
            The size for the zero padding on two sides.
        fill_value : Union[int, Sequence], optional, default=127
            The value(s) to fill for padding or cutout.
        crop_size : int, optional, default=0
            The size for random-or-center cropping.
        random_crop_size: int, optional, default=0
            The size for sampling-based random cropping.
        cutout_size : int, optional, default=0
            The square size for the cutout algorithm.
        mirror : bool, optional, default=False
            Whether to apply the mirror (flip horizontally).
        random_scales : Sequence[float], optional, default=(0.08, 1.)
            The range of scales to sample a crop randomly.
        random_aspect_ratios : Sequence[float], optional, default=(0.75, 1.33)
            The range of aspect ratios to sample a crop randomly.
        distort_color : bool, optional, default=False
            Whether to apply color distortion.
        inverse_color : bool, option, default=False
            Whether to inverse channels for color images.
        training : optional, default=True
            Whether to enable the training randoms.
        batch_size : int, optional, default=128
            The size of a mini-batch.
        prefetch_depth : int, optional, default=4
            The number of prefetching queues.
        num_transformers : int, optional, default=-1
            The number of transformers to process image.
        seed : int, optional
            The random seed to use instead.

        """
        super(DataIterator, self).__init__(daemon=True)
        # Distributed settings.
        rank, group_size = 0, 1
        process_group = distributed.get_group()
        if process_group is not None and kwargs.get('training', True):
            group_size = process_group.size
            rank = distributed.get_rank(process_group)

        # Configuration.
        self._prefetch_depth = kwargs.get('prefetch_depth', 4)
        self._num_readers = kwargs.get('num_readers', 1)
        self._num_workers = kwargs.get('num_workers', -1)
        self._batch_size = kwargs.get('batch_size', 128)

        # Io-Aware Policy.
        if self._num_workers == -1:
            self._num_workers = 1
            # Add a transformer for cropping.
            if kwargs.get('random_crop_size', 0) > 0:
                self._num_workers += 1
            # Add a transformer for distortion.
            if kwargs.get('distort_color', False):
                self._num_workers += 1

        # Initialize queues.
        num_batches = self._prefetch_depth * self._num_readers
        self._reader_queue = mp.Queue(num_batches * self._batch_size)
        self._worker_queue = mp.Queue(num_batches * self._batch_size)
        self._batch_queue = queue.Queue(num_batches)

        # Initialize readers.
        self._readers = []
        for i in range(self._num_readers):
            part_idx, num_parts = i, self._num_readers
            num_parts *= group_size
            part_idx += rank * self._num_readers
            self._readers.append(reader.DataReader(
                part_idx=part_idx, num_parts=num_parts, **kwargs))
            self._readers[i]._seed += part_idx
            self._readers[i]._reader_queue = self._reader_queue
            self._readers[i].start()
            time.sleep(0.1)

        # Initialize transformers.
        self._workers = []
        for i in range(self._num_workers):
            p = data_worker.DataWorker(**kwargs)
            p._seed += (i + rank * self._num_workers)
            p._reader_queue = self._reader_queue
            p._worker_queue = self._worker_queue
            p.start()
            self._workers.append(p)
            time.sleep(0.1)

        # Register cleanup callbacks.
        def cleanup():
            def terminate(processes):
                for p in processes:
                    p.terminate()
                    p.join()
            terminate(self._workers)
            if rank == 0:
                logging.info('Terminate DataWorker.')
            terminate(self._readers)
            if rank == 0:
                logging.info('Terminate DataReader.')
        import atexit
        atexit.register(cleanup)

        # Start batch prefetching.
        self.start()

    def next(self):
        """Return the next batch of data."""
        return self.__next__()

    def run(self):
        while True:
            inputs = self._worker_queue.get()
            outputs = []
            for data in inputs:
                output_shape = (self._batch_size,) + data.shape[:]
                buffer = numpy.empty(output_shape, dtype=data.dtype)
                buffer[0] = data
                outputs.append(buffer)
            for i in range(1, self._batch_size):
                inputs = self._worker_queue.get()
                for j, input in enumerate(inputs):
                    outputs[j][i] = input
            self._batch_queue.put(outputs)

    def __iter__(self):
        """Return the iterator self."""
        return self

    def __next__(self):
        """Return the next batch of data."""
        return self._batch_queue.get()
