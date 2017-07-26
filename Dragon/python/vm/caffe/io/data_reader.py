# --------------------------------------------------------
# Caffe for Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import math
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
from dragon.tools.db import LMDB

from __init__ import GetProperty

class DataReader(Process):
    def __init__(self, **kwargs):
        super(DataReader, self).__init__()
        self._source = GetProperty(kwargs, 'source', '')
        self._use_shuffle = GetProperty(kwargs, 'shuffle', False)
        self._use_step = GetProperty(kwargs, 'node_step', False)
        self._chunk_size = GetProperty(kwargs, 'chunk_size', 4) # >=4MB

        self._num_parts = 1
        self._part_idx = 0
        self._random_seed = config.GetRandomSeed()

        self._cur_idx = 0
        self._cur_chunk_idx = 0

        self.Q_out = None
        self.daemon = True

        def cleanup():
            print 'Terminating DataReader......'
            self.terminate()
            self.join()
        import atexit
        atexit.register(cleanup)

    def element(self):
        return self._db.value()

    def reset(self):
        if self._use_shuffle:
            self._cur_chunk_idx = 0
            self._perm = npr.permutation(self._num_shuffle_parts)
            self._start_idx = self._part_idx * self._num_shuffle_parts + self._perm[self._cur_chunk_idx]
            self._start_idx = self._start_idx * self._chunk_size
            if self._start_idx >= self._db_size: self.next_chunk()
            self._end_idx = self._start_idx + self._chunk_size
            self._end_idx = min(self._db_size, self._end_idx)
            #self._part_idx = (self._part_idx + 1) % self._num_parts  # fast hard disk driver is required

        elif self._use_step:
            self._start_idx = self._part_idx * self._epoch_size
            self._end_idx = self._start_idx + self._epoch_size
            self._end_idx = min(self._db_size, self._end_idx)
            #self._part_idx = (self._part_idx + 1) % self._num_parts  # fast hard disk driver is required
        else:
            self._start_idx = 0
            self._end_idx = self._db_size

        self._cur_idx = self._start_idx
        self._db.set(str(self._cur_idx).zfill(self._db_zfill))

    def next_record(self):
        self._cur_idx += 1
        self._db.next()

    def next_chunk(self):
        self._cur_chunk_idx += 1
        if self._cur_chunk_idx >= self._num_shuffle_parts: self.reset()
        else:
            self._start_idx = self._part_idx * self._num_shuffle_parts + self._perm[self._cur_chunk_idx]
            self._start_idx = self._start_idx * self._chunk_size
            if self._start_idx >= self._db_size: self.next_chunk()
            else:
                self._end_idx = self._start_idx + self._chunk_size
                self._end_idx = min(self._db_size, self._end_idx)
            self._cur_idx = self._start_idx
            self._db.set(str(self._cur_idx).zfill(self._db_zfill))

    def run(self):
        # fix seed
        npr.seed(self._random_seed)

        # init db
        self._db = LMDB()
        self._db.open(self._source)
        self._db_size = int(self._db.get('size'))
        self._db_zfill = int(self._db.get('zfill'))
        self._epoch_size = self._db_size / self._num_parts + 1
        self._num_shuffle_parts = int(math.ceil(self._db._total_size * 1.1 /
                                               (self._num_parts * self._chunk_size << 20)))
        self._chunk_size = self._db_size / self._num_shuffle_parts / self._num_parts + 1

        # init env
        self.reset()

        # run !
        while True:
            self.Q_out.put(self.element())
            self.next_record()
            if self._cur_idx >= self._end_idx:
                if self._use_shuffle: self.next_chunk()
                else: self.reset()

