# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import time
import pprint
from multiprocessing import Queue
from six.moves import range as xrange

import dragon.core.mpi as mpi
from dragon.config import logger

from .data_reader import DataReader
from .data_transformer import DataTransformer
from .blob_fetcher import BlobFetcher

from .utils import GetProperty

class DataBatch(object):
    """
    DataBatch aims to prefetch data by ``Triple-Buffering``.

    It takes full advantages of the Process/Thread of Python,

    which provides remarkable I/O speed up for scalable distributed training.
    """
    def __init__(self, **kwargs):
        """Construct a ``DataBatch``.

        Parameters
        ----------
        source : str
            The path of database.
        shuffle : boolean
            Whether to shuffle the data.
        node_step: boolean
            Whether to split data for multiple parallel nodes.
        num_chunks : int
            The number of chunks to split. Default is ``2048``.
        chunk_size : int
            The size(MB) of each chunk. Default is -1 (Refer ``num_chunks``).
        mean_values : list
            The mean value of each image channel.
        scale : float
            The scale performed after mean subtraction. Default is ``1.0``.
        padding : int
            The zero-padding size. Default is ``0`` (Disabled).
        fill_value : int
            The value to fill when padding is valid. Default is ``127``.
        crop_size : int
            The crop size. Default is ``0`` (Disabled).
        mirror : boolean
            Whether to flip(horizontally) images. Default is ``False``.
        color_augmentation : boolean
            Whether to distort colors. Default is ``False``.
        min_random_scale : float
            The min scale of the input images. Default is ``1.0``.
        max_random_scale : float
            The max scale of the input images. Default is ``1.0``.
        force_color : boolean
            Set to duplicate channels for gray. Default is ``False``.
        phase : str
            The phase of this operator, ``TRAIN`` or ``TEST``. Default is ``TRAIN``.
        batch_size : int
            The size of a training batch.
        partition : boolean
            Whether to partition batch. Default is ``False``.
        prefetch : int
            The prefetch count. Default is ``5``.

        """
        super(DataBatch, self).__init__()
        # init mpi
        global_rank = 0; local_rank = 0; group_size = 1
        if mpi.Is_Init():
            idx, group = mpi.AllowParallel()
            if idx != -1:  # data parallel
                global_rank = mpi.Rank()
                group_size = len(group)
                for i, node in enumerate(group):
                    if global_rank == node: local_rank = i
        kwargs['group_size'] = group_size

        # configuration
        self._prefetch = GetProperty(kwargs, 'prefetch', 5)
        self._num_readers = GetProperty(kwargs, 'num_readers', 1)
        self._num_transformers = GetProperty(kwargs, 'num_transformers', -1)
        self._max_transformers = GetProperty(kwargs, 'max_transformers', 3)
        self._num_fetchers = GetProperty(kwargs, 'num_fetchers', 1)

        # io-aware policy
        if self._num_transformers == -1:
            self._num_transformers = 1
            # add 1 transformer for color augmentation
            if GetProperty(kwargs, 'color_augmentation', False):
                self._num_transformers += 1
            # add 1 transformer for random scale
            if GetProperty(kwargs, 'max_random_scale', 1.0) - \
                    GetProperty(kwargs, 'min_random_scale', 1.0) != 0:
                self._num_transformers += 1
        self._num_transformers = min(self._num_transformers, self._max_transformers)

        self._batch_size = GetProperty(kwargs, 'batch_size', 100)
        self._partition = GetProperty(kwargs, 'partition', False)
        if self._partition:
            self._batch_size = int(self._batch_size / kwargs['group_size'])

        # init queues
        self.Q_level_1 = Queue(self._prefetch * self._num_readers * self._batch_size)
        self.Q_level_2 = Queue(self._prefetch * self._num_readers * self._batch_size)
        self.Q_level_3 = Queue(self._prefetch * self._num_readers)

        # init readers
        self._readers = []
        for i in xrange(self._num_readers):
            self._readers.append(DataReader(**kwargs))
            self._readers[-1].Q_out = self.Q_level_1

        for i in xrange(self._num_readers):
            num_parts = self._num_readers
            part_idx = i

            if self._readers[i]._use_shuffle \
                    or self._readers[i]._use_step:
                num_parts *= group_size
                part_idx += local_rank * self._num_readers

            self._readers[i]._num_parts = num_parts
            self._readers[i]._part_idx = part_idx
            self._readers[i]._random_seed += part_idx
            self._readers[i].start()
            time.sleep(0.1)

        # init transformers
        self._transformers = []
        for i in xrange(self._num_transformers):
            transformer = DataTransformer(**kwargs)
            transformer._random_seed += (i + local_rank * self._num_transformers)
            transformer.Q_in = self.Q_level_1
            transformer.Q_out = self.Q_level_2
            transformer.start()
            self._transformers.append(transformer)
            time.sleep(0.1)

        # init blob fetchers
        self._fetchers = []
        for i in xrange(self._num_fetchers):
            fetcher = BlobFetcher(**kwargs)
            fetcher.Q_in = self.Q_level_2
            fetcher.Q_out = self.Q_level_3
            fetcher.start()
            self._fetchers.append(fetcher)
            time.sleep(0.1)

        self.echo()

    def get(self):
        """Get a batch.

        Returns
        -------
        tuple
            The batch, representing data and labels respectively.

        """
        return self.Q_level_3.get()

    def echo(self):
        """
        Print I/O Information.
        """
        logger.info('---------------------------------------------------------')
        logger.info('BatchReader, Using config:')
        params = {'prefetching': self._prefetch,
                  'num_readers': self._num_readers,
                  'num_transformers': self._num_transformers,
                  'num_fetchers': self._num_fetchers}
        pprint.pprint(params)
        logger.info('---------------------------------------------------------')
