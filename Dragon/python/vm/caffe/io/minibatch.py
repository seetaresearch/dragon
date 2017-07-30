# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import time
import pprint
from multiprocessing import Queue

import dragon.core.mpi as mpi

from data_reader import DataReader
from data_transformer import DataTransformer
from blob_fetcher import BlobFetcher
from __init__ import GetProperty

class DataBatch(object):
    def __init__(self, **kwargs):

        """DataBatch use Triple-Buffering to speed up"""

        # init mpi
        global_rank = 0; local_rank = 0; group_size = 1
        if mpi.is_init():
            idx, group = mpi.allow_parallel()
            if idx != -1:  # data parallel
                global_rank = mpi.rank()
                group_size = len(group)
                for i, node in enumerate(group):
                    if global_rank == node: local_rank = i
        kwargs['group_size'] = group_size

        # configuration
        self._prefetch = GetProperty(kwargs, 'prefetch', 40)
        self._num_readers = GetProperty(kwargs, 'num_readers', 1)
        self._num_transformers = GetProperty(kwargs, 'num_transformers', -1)
        self._num_fetchers = GetProperty(kwargs, 'num_fetchers', 3)

        # default policy
        if self._num_transformers == -1:
            self._num_transformers = 1
            # add 1 transformer for color augmentation
            if GetProperty(kwargs, 'color_augmentation', False):
                self._num_transformers += 1
            # add 1 transformer for random scale
            if GetProperty(kwargs, 'max_random_scale', 1.0) - \
                GetProperty(kwargs, 'min_random_scale', 1.0) != 0:
                self._num_transformers +=1

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

        #self.echo()

    @property
    def blobs(self):
        return self.Q_level_3.get()

    def echo(self):
        print '---------------------------------------------------------'
        print 'BatchReader, Using config:'
        params = {'prefetching': self._prefetch,
                  'num_readers': self._num_readers,
                  'num_transformers': self._num_transformers,
                  'num_fetchers': self._num_fetchers}
        pprint.pprint(params)
        print '---------------------------------------------------------'
