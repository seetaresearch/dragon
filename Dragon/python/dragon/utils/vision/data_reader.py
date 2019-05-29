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

import math
import numpy
import multiprocessing

from dragon import config as _cfg
from dragon.tools import db as _db


class DataReader(multiprocessing.Process):
    """DataReader is deployed to queue encoded str from `LMDB`_.

    It is supported to adaptively partition and shuffle records over all distributed nodes.

    """
    def __init__(self, **kwargs):
        """Construct a ``DataReader``.

        Parameters
        ----------
        source : str
            The path of database.
        shuffle : bool, optional, default=False
            Whether to shuffle the data.
        num_chunks : int, optional, default=2048
            The number of chunks to split.

        """
        super(DataReader, self).__init__()
        self._source = kwargs.get('source', '')
        self._use_shuffle = kwargs.get('shuffle', False)
        self._num_chunks = kwargs.get('num_chunks', 2048)
        self._part_idx, self._num_parts = 0, 1
        self._cursor, self._chunk_cursor = 0, 0
        self._rng_seed = _cfg.GetRandomSeed()
        self.Q_out = None
        self.daemon = True

    def element(self):
        """Get the value of current record.

        Returns
        -------
        str
            The encoded str.

        """
        return self._db.value()

    def redirect(self, target):
        """Redirect to the target position.

        Parameters
        ----------
        target : int
            The key of the record.

        Returns
        -------
        None

        Notes
        -----
        The redirection reopens the database.

        You can drop caches by ``echo 3 > /proc/sys/vm/drop_caches``.

        This will disturb getting stuck when *Database Size* >> *RAM Size*.

        """
        self._db.close()
        self._db.open(self._source)
        self._cursor = target
        self._db.set(str(target).zfill(self._zfill))

    def reset(self):
        """Reset the cursor and environment.

        Returns
        -------
        None

        """
        if self._num_parts > 1 or self._use_shuffle:
            self._chunk_cursor = 0
            self._part_idx = (self._part_idx + 1) % self._num_parts
            if self._use_shuffle: self._perm = numpy.random.permutation(self._perm_size)
            self._head = self._part_idx * self._perm_size + self._perm[self._chunk_cursor]
            self._tail = self._head * self._chunk_size
            if self._head >= self._num_entries: self.next_chunk()
            self._tail = self._head + self._chunk_size
            self._tail = min(self._num_entries, self._tail)
        else:
            self._head, self._tail = 0, self._num_entries
        self.redirect(self._head)

    def next_record(self):
        """Step the cursor of records.

        Returns
        -------
        None

        """
        self._db.next()
        self._cursor += 1

    def next_chunk(self):
        """Step the cursor of chunks.

        Returns
        -------
        None

        """
        self._chunk_cursor += 1
        if self._chunk_cursor >= self._perm_size: self.reset()
        else:
            self._head = self._part_idx * self._perm_size + self._perm[self._chunk_cursor]
            self._head = self._head * self._chunk_size
            if self._head >= self._num_entries:
                self.next_chunk()
            else:
                self._tail = self._head + self._chunk_size
                self._tail = min(self._num_entries, self._tail)
            self.redirect(self._head)

    def run(self):
        """Start the process.

        Returns
        -------
        None

        """
        # Fix seed
        numpy.random.seed(self._rng_seed)

        # Init db
        self._db = _db.LMDB()
        self._db.open(self._source)
        self._zfill = self._db.zfill()
        self._num_entries = self._db.num_entries()

        epoch_size = self._num_entries // self._num_parts + 1

        if self._use_shuffle:
            if self._num_chunks <= 0:
                # Each chunk has at most 1 record (Record-Wise)
                self._chunk_size, self._perm_size = 1, epoch_size
            else:
                # Search a optimal chunk size (Chunk-Wise)
                min_size, max_size = \
                    1, self._db._total_size * 1.0 \
                        / (self._num_chunks * (1 << 20))
                while min_size * 2 < max_size: min_size *= 2
                self._perm_size = int(math.ceil(
                    self._db._total_size * 1.1 /
                        (self._num_parts * min_size << 20)))
                self._chunk_size = int(
                    self._num_entries * 1.0 /
                        (self._perm_size * self._num_parts) + 1)
                limit = (self._num_parts - 0.5) * self._perm_size * self._chunk_size
                if self._num_entries <= limit:
                    # Roll back to Record-Wise shuffle
                    self._chunk_size, self._perm_size = 1, epoch_size
        else:
            # One chunk has at most K records
            self._chunk_size, self._perm_size = epoch_size, 1

        self._perm = numpy.arange(self._perm_size)

        # Init env
        self.reset()

        # Run!
        while True:
            self.Q_out.put(self.element())
            self.next_record()
            if self._cursor >= self._tail:
                if self._num_parts > 1 or self._use_shuffle:
                    self.next_chunk()
                else: self.reset()