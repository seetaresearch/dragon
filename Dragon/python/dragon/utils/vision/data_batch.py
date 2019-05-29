# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#      <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time

from multiprocessing import Queue
from dragon.core import mpi as _mpi
from dragon.core import logging as _logging

from .data_reader import DataReader
from .data_transformer import DataTransformer
from .blob_fetcher import BlobFetcher


class DataBatch(object):
    """DataBatch aims to prefetch data by *Triple-Buffering*.

    It takes full advantages of the Process/Thread of Python,

    which provides remarkable I/O speed up for scalable distributed training.

    """
    def __init__(self, **kwargs):
        """Construct a ``DataBatch``.

        Parameters
        ----------
        source : str
            The path of database.
        shuffle : bool, optional, default=False
            Whether to shuffle the data.
        num_chunks : int, optional, default=2048
            The number of chunks to split.
        padding : int, optional, default=0
            The zero-padding size.
        fill_value : int or sequence, optional, default=127
            The value(s) to fill for padding or cutout.
        crop_size : int, optional, default=0
            The cropping size.
        cutout_size : int, optional, default=0
            The square size to cutout.
        mirror : bool, optional, default=False
            Whether to mirror(flip horizontally) images.
        color_augmentation : bool, optional, default=False
            Whether to use color distortion.1
        min_random_scale : float, optional, default=1.
            The min scale of the input images.
        max_random_scale : float, optional, default=1.
            The max scale of the input images.
        force_gray : bool, optional, default=False
            Set not to duplicate channel for gray.
        phase : {'TRAIN', 'TEST'}, optional
            The optional running phase.
        batch_size : int, optional, default=128
            The size of a mini-batch.
        partition : bool, optional, default=False
            Whether to partition batch for parallelism.
        prefetch : int, optional, default=5
            The prefetch count.

        """
        super(DataBatch, self).__init__()
        # Init mpi
        global_rank, local_rank, group_size = 0, 0, 1
        if _mpi.Is_Init() and kwargs.get(
                'phase', 'TRAIN') == 'TRAIN':
            rank, group = _mpi.AllowParallel()
            if rank != -1: # DataParallel
                global_rank, group_size = _mpi.Rank(), len(group)
                for i, node in enumerate(group):
                    if global_rank == node: local_rank = i
        kwargs['group_size'] = group_size

        # Configuration
        self._prefetch = kwargs.get('prefetch', 5)
        self._num_readers = kwargs.get('num_readers', 1)
        self._num_transformers = kwargs.get('num_transformers', -1)
        self._max_transformers = kwargs.get('max_transformers', 3)
        self._num_fetchers = kwargs.get('num_fetchers', 1)

        # Io-Aware Policy
        if self._num_transformers == -1:
            self._num_transformers = 1
            # Add 1 transformer for color augmentation
            if kwargs.get('color_augmentation', False):
                self._num_transformers += 1
            # Add 1 transformer for random scale
            if kwargs.get('max_random_scale', 1.0) - \
                kwargs.get('min_random_scale', 1.0) != 0:
                    self._num_transformers += 1
            # Add 1 transformer for random crop
            if kwargs.get('crop_size', 0) > 0 and \
                kwargs.get('phase', 'TRAIN') == 'TRAIN':
                    self._num_transformers += 1
        self._num_transformers = min(
            self._num_transformers, self._max_transformers)

        self._batch_size = kwargs.get('batch_size', 128)
        self._partition = kwargs.get('partition', False)
        if self._partition: self._batch_size //= kwargs['group_size']

        # Init queues
        self.Q1 = Queue(self._prefetch * self._num_readers * self._batch_size)
        self.Q2 = Queue(self._prefetch * self._num_readers * self._batch_size)
        self.Q3 = Queue(self._prefetch * self._num_readers)

        # Init readers
        self._readers = []
        for i in range(self._num_readers):
            self._readers.append(DataReader(**kwargs))
            self._readers[-1].Q_out = self.Q1

        for i in range(self._num_readers):
            part_idx, num_parts = i, self._num_readers
            num_parts *= group_size
            part_idx += local_rank * self._num_readers
            self._readers[i]._num_parts = num_parts
            self._readers[i]._part_idx = part_idx
            self._readers[i]._rng_seed += part_idx
            self._readers[i].start()
            time.sleep(0.1)

        # Init transformers
        self._transformers = []
        for i in range(self._num_transformers):
            transformer = DataTransformer(**kwargs)
            transformer._rng_seed += (i + local_rank * self._num_transformers)
            transformer.Q_in, transformer.Q_out = self.Q1, self.Q2
            transformer.start()
            self._transformers.append(transformer)
            time.sleep(0.1)

        # Init blob fetchers
        self._fetchers = []
        for i in range(self._num_fetchers):
            fetcher = BlobFetcher(**kwargs)
            fetcher.Q_in, fetcher.Q_out = self.Q2, self.Q3
            fetcher.start()
            self._fetchers.append(fetcher)
            time.sleep(0.1)

        def cleanup():
            def terminate(processes):
                for process in processes:
                    process.terminate()
                    process.join()
            terminate(self._fetchers)
            if local_rank == 0: _logging.info('Terminate BlobFetcher.')
            terminate(self._transformers)
            if local_rank == 0: _logging.info('Terminate DataTransformer.')
            terminate(self._readers)
            if local_rank == 0: _logging.info('Terminate DataReader.')
        import atexit
        atexit.register(cleanup)

    def get(self):
        """Get a batch.

        Returns
        -------
        tuple
            The batch, representing data and labels respectively.

        """
        return self.Q3.get()