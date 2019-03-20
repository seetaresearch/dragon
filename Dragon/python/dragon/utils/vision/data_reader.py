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
import numpy as np
import numpy.random as npr
from multiprocessing import Process

import dragon.config as config
from dragon.tools.db import LMDB


class DataReader(Process):
    """DataReader is deployed to queue encoded str from `LMDB`_.

    It is supported to adaptively partition and shuffle records over all distributed nodes.

    """
    def __init__(self, **kwargs):
        """Construct a ``DataReader``.

        Parameters
        ----------
        source : str
            The path of database.
        multiple_nodes: boolean, optional, default=False
            Whether to split data for multiple parallel nodes.
        shuffle : bool, optional, default=False
            Whether to shuffle the data.
        num_chunks : int, optional, default=2048
            The number of chunks to split.
        chunk_size : int, optional, default=-1
            The size(MB) of each chunk.

        """
        super(DataReader, self).__init__()
        self._source = kwargs.get('source', '')
        self._multiple_nodes = kwargs.get('multiple_nodes', False)
        self._use_shuffle = kwargs.get('shuffle', False)
        self._use_instance_chunk = kwargs.get('instance_chunk', False)
        self._num_chunks = kwargs.get('num_chunks', 2048)
        self._chunk_size = kwargs.get('chunk_size', -1)

        self._part_idx, self._num_parts = 0, 1
        self._cur_idx, self._cur_chunk_idx = 0, 0
        self._random_seed = config.GetRandomSeed()

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
        self._db.set(str(self._cur_idx).zfill(self._zfill))

    def reset(self):
        """Reset the cursor and environment.

        Returns
        -------
        None

        """
        if self._multiple_nodes or self._use_shuffle:
            if self._use_shuffle: self._perm = npr.permutation(self._num_shuffle_parts)
            self._cur_chunk_idx = 0
            self._start_idx = int(self._part_idx * self._num_shuffle_parts + self._perm[self._cur_chunk_idx])
            self._start_idx = int(self._start_idx * self._chunk_size)
            if self._start_idx >= self._num_entries: self.next_chunk()
            self._end_idx = self._start_idx + self._chunk_size
            self._end_idx = min(self._num_entries, self._end_idx)
        else:
            self._start_idx = 0
            self._end_idx = self._num_entries

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
            if self._start_idx >= self._num_entries: self.next_chunk()
            else:
                self._end_idx = self._start_idx + self._chunk_size
                self._end_idx = min(self._num_entries, self._end_idx)
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
        self._zfill = self._db.zfill()
        self._num_entries = self._db.num_entries()
        self._epoch_size = int(self._num_entries/ self._num_parts + 1)

        if self._use_shuffle:
            if self._chunk_size == 1:
                # Each chunk has at most 1 record [For Fully Shuffle]
                self._chunk_size, self._num_shuffle_parts = \
                    1, int(self._num_entries / self._num_parts) + 1
            else:
                if self._use_shuffle and self._chunk_size == -1:
                    # Search a optimal chunk size by chunks [For Chunk Shuffle]
                    max_chunk_size = self._db._total_size / ((self._num_chunks * (1 << 20)))
                    min_chunk_size = 1
                    while min_chunk_size * 2 < max_chunk_size: min_chunk_size *= 2
                    self._chunk_size = min_chunk_size
                    self._num_shuffle_parts = int(math.ceil(self._db._total_size * 1.1 /
                                                 (self._num_parts * self._chunk_size << 20)))
                    self._chunk_size = int(self._num_entries / self._num_shuffle_parts / self._num_parts + 1)
                    limit = (self._num_parts - 0.5) * self._num_shuffle_parts * self._chunk_size
                    if self._num_entries <= limit:
                        # Roll back to fully shuffle
                        self._chunk_size, self._num_shuffle_parts = \
                            1, int(self._num_entries / self._num_parts) + 1
        else:
            # Each chunk has at most K records [For Multiple Nodes]
            # Note that if ``shuffle`` and ``multiple_nodes`` are all ``False``,
            # ``chunk_size`` and ``num_shuffle_parts`` are meaningless
            self._chunk_size = int(self._num_entries / self._num_parts) + 1
            self._num_shuffle_parts = 1

        self._perm = np.arange(self._num_shuffle_parts)

        # Init env
        self.reset()

        # Run!
        while True:
            self.Q_out.put(self.element())
            self.next_record()
            if self._cur_idx >= self._end_idx:
                if self._multiple_nodes or \
                    self._use_shuffle: self.next_chunk()
                else: self.reset()