# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import lmdb
import os
import sys

def wrapper_str(raw_str):
    if sys.version_info >= (3, 0):
        return raw_str.encode()
    return raw_str

class LMDB(object):
    def __init__(self, max_commit=10000):
        self._max_commit = max_commit
        self._cur_put = 0
        self._total_size = 0
        self._buffer = []


    def open(self, database_path, mode='r'):
        if mode == 'r':
            assert os.path.exists(database_path), 'database path is not exist'
            self.env = lmdb.open(database_path, readonly=True, lock=False)
            self._total_size = self.env.info()['map_size']
        if mode == 'w':
            assert not os.path.isdir(database_path), 'database path is not invalid'
            self.env = lmdb.open(database_path, writemap=True)
        self.txn = self.env.begin(write=(mode == 'w'))
        self.cursor = self.txn.cursor()


    def _try_put(self):
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
        self._buffer.append((wrapper_str(key), wrapper_str(value)))
        self._cur_put += 1
        if (self._cur_put >= self._max_commit): self._try_put()


    def commit(self):
        self._try_put()
        self.txn.commit()
        self.txn = self.env.begin(write=True)


    def set(self, key):
        self.cursor.set_key(wrapper_str(key))


    def get(self, key):
        cursor = self.txn.cursor()
        return cursor.get(wrapper_str(key))


    def next(self):
        if not self.cursor.next():
            self.cursor.first()
        if self.key() == 'size' or self.key() == 'zfill':
            self.next()


    def key(self):
        return self.cursor.key()


    def value(self):
        return self.cursor.value()


    def close(self):
        self.env.close()