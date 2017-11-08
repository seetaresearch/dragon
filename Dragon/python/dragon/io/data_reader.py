# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import math
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
from dragon.tools.db import LMDB

from .utils import GetProperty

class DataReader(Process):
    """
    DataReader is deployed to queue encoded str from `LMDB`_.

    It is supported to adaptively partition and shuffle records over all distributed nodes.
    """
    def __init__(self, **kwargs):
        """Construct a ``DataReader``.

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

        """
        super(DataReader, self).__init__()
        self._source = GetProperty(kwargs, 'source', '')
        self._use_shuffle = GetProperty(kwargs, 'shuffle', False)
        self._use_step = GetProperty(kwargs, 'node_step', False)
        self._num_chunks = GetProperty(kwargs, 'num_chunks', 2048)
        self._chunk_size = GetProperty(kwargs, 'chunk_size', -1)

        self._num_parts = 1
        self._part_idx = 0
        self._random_seed = config.GetRandomSeed()

        self._cur_idx = 0
        self._cur_chunk_idx = 0

        self.Q_out = None
        self.daemon = True

        def cleanup():
            from dragon.config import logger
            logger.info('Terminating DataReader......')
            self.terminate()
            self.join()
        import atexit
        atexit.register(cleanup)

    def element(self):
        """Get the value of current record.

        Returns
        -------
        str
            The encoded str.

        """
        return self._db.value()

    def redirect(self, target_idx):
        """Redirect to the target position.

        Parameters
        ----------
        target_idx : int
            The key of instance in ``LMDB``.

        Returns
        -------
        None

        Notes
        -----
        The redirection reopens the ``LMDB``.

        You can drop caches by ``echo 3 > /proc/sys/vm/drop_caches``.

        This will disturb getting stuck when ``Database Size`` >> ``RAM Size``.

        """
        self._db.close()
        self._db.open(self._source)
        self._cur_idx = target_idx
        self._db.set(str(self._cur_idx).zfill(self._db_zfill))

    def reset(self):
        """Reset the cursor and environment.

        Returns
        -------
        None

        """
        if self._use_shuffle or self._use_step:
            if self._use_shuffle:
                self._perm = npr.permutation(self._num_shuffle_parts)
            self._cur_chunk_idx = 0
            self._start_idx = int(self._part_idx * self._num_shuffle_parts + self._perm[self._cur_chunk_idx])
            self._start_idx = int(self._start_idx * self._chunk_size)
            if self._start_idx >= self._db_size: self.next_chunk()
            self._end_idx = self._start_idx + self._chunk_size
            self._end_idx = min(self._db_size, self._end_idx)
        else:
            self._start_idx = 0
            self._end_idx = self._db_size

        self.redirect(self._start_idx)

    def next_record(self):
        """Step the cursor of records.

        Returns
        -------
        None

        """
        self._cur_idx += 1
        self._db.next()

    def next_chunk(self):
        """Step the cursor of shuffling chunks.

        Returns
        -------
        None

        """
        self._cur_chunk_idx += 1
        if self._cur_chunk_idx >= self._num_shuffle_parts: self.reset()
        else:
            self._start_idx = self._part_idx * self._num_shuffle_parts + self._perm[self._cur_chunk_idx]
            self._start_idx = self._start_idx * self._chunk_size
            if self._start_idx >= self._db_size: self.next_chunk()
            else:
                self._end_idx = self._start_idx + self._chunk_size
                self._end_idx = min(self._db_size, self._end_idx)
            self.redirect(self._start_idx)

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        # fix seed
        npr.seed(self._random_seed)

        # init db
        self._db = LMDB()
        self._db.open(self._source)
        self._db_size = int(self._db.get('size'))
        self._db_zfill = int(self._db.get('zfill'))
        self._epoch_size = int(self._db_size / self._num_parts + 1)
        # search a optimal chunk size by chunks
        if self._chunk_size == -1:
            max_chunk_size = self._db._total_size / ((self._num_chunks * (1 << 20)))
            min_chunk_size = 1
            while min_chunk_size * 2 < max_chunk_size: min_chunk_size *= 2
            self._chunk_size = min_chunk_size
        self._num_shuffle_parts = int(math.ceil(self._db._total_size * 1.1 /
                                               (self._num_parts * self._chunk_size << 20)))
        self._chunk_size = int(self._db_size / self._num_shuffle_parts / self._num_parts + 1)
        self._perm = npr.permutation(self._num_shuffle_parts)

        # init env
        self.reset()

        # run
        while True:
            self.Q_out.put(self.element())
            self.next_record()
            if self._cur_idx >= self._end_idx:
                if self._use_shuffle or self._use_step: self.next_chunk()
                else: self.reset()