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

import os
import sys
import lmdb


def wrapper_str(raw_str):
    if sys.version_info >= (3, 0):
        return raw_str.encode()
    return raw_str


class LMDB(object):
    """A wrapper of ``LMDB`` package.

    We exploit more Key-Value specifics than Caffe's naive usages,

    such as: ``Distributed Prefetching``, ``Chunk Shuffling``, and ``Cache Dropping``,

    which provides splendid I/O efficiency comparing to ``MXNet`` or ``TensorFlow``.

    Examples
    --------
    >>> db = LMDB()
    >>> db.open('/xxx/yyy_lmdb', mode='w')
    >>> db.put('000001', 'A')
    >>> db.commit()
    >>> db.close()

    >>> db = LMDB()
    >>> db.open('/xxx/yyy_lmdb', mode='r')
    >>> db.set('000001')
    >>> print(db.value())
    >>> 'A'

    """
    def __init__(self, max_commit=10000):
        """Construct a ``LMDB``.

        Parameters
        ----------
        max_commit : int
            The max buffer size before committing automatically.

        Returns
        -------
        LMDB
            The database instance.

        """
        self._max_commit = max_commit
        self._cur_put = 0
        self._total_size = 0
        self._buffer = []

    def open(self, database_path, mode='r'):
        """Open the database.

        Parameters
        ----------
        database_path : str
            The path of the LMDB database.
        mode : str
            The mode. ``r`` or ``w``.

        Returns
        -------
        None

        """
        if mode == 'r':
            assert os.path.exists(database_path), 'database path is not exist'
            self.env = lmdb.open(database_path, readonly=True, lock=False)
            self._total_size = self.env.info()['map_size']
        if mode == 'w':
            self.env = lmdb.open(database_path, writemap=True)
        self.txn = self.env.begin(write=(mode == 'w'))
        self.cursor = self.txn.cursor()

    def zfill(self):
        self.cursor.first()
        return len(self.key())

    def num_entries(self):
        return self.env.stat()['entries']

    def _try_put(self):
        """Try to commit the buffers.

        This is a trick to prevent ``1TB`` disk space required on ``NTFS`` partition.

        Returns
        -------
        None

        """
        for pair in self._buffer:
            key, value = pair
            try: self.txn.put(key, value)
            except lmdb.MapFullError as e:
                new_size = self.env.info()['map_size'] * 2
                print('doubling LMDB map size to %d MB' % (new_size >> 20))
                self.txn.abort()
                self.env.set_mapsize(new_size)
                self.txn = self.env.begin(write=True)
                self._try_put()
        self._cur_put = 0
        self._buffer = []

    def put(self, key, value):
        """Put the item.

        Parameters
        ----------
        key : str
            The key.
        value : str
            The value.

        Returns
        -------
        None

        """
        self._buffer.append((wrapper_str(key), value))
        self._cur_put += 1
        if (self._cur_put >= self._max_commit): self._try_put()

    def commit(self):
        """Commit all items that have been put.

        Returns
        -------
        None

        """
        self._try_put()
        self.txn.commit()
        self.txn = self.env.begin(write=True)

    def set(self, key):
        """Set the cursor to the specific key.

        Parameters
        ----------
        key : str
            The key to set.

        Returns
        -------
        None

        """
        self.cursor.set_key(wrapper_str(key))

    def get(self, key):
        """Get the value of the specific key.

        Parameters
        ----------
        key : str
            The key.

        Returns
        -------
        str
            The value.

        """
        cursor = self.txn.cursor()
        return cursor.get(wrapper_str(key))

    def next(self):
        """Set the cursor to the next.

        Returns
        -------
        None

        """
        if not self.cursor.next():
            self.cursor.first()
        if self.key() == 'size' or self.key() == 'zfill':
            self.next()

    def key(self):
        """Get the key under the current cursor.

        Returns
        -------
        str
            The key.

        """
        return self.cursor.key()

    def value(self):
        """Get the value under the current cursor.

        Returns
        -------
        str
            The value.

        """
        return self.cursor.value()

    def close(self):
        """Close the database.

        Returns
        -------
        None

        """
        self.env.close()