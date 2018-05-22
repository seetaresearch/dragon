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

from dragon.vm.torch.utils.data.io.data_reader import DataReader
from dragon.vm.torch.utils.data.io.data_transformer import DataTransformer


class _LMDBStream(object):
    def __init__(self, database, transform, color_space='RGB'):
        from multiprocessing import Queue
        self.Q = Queue(1)
        # Create a DataReader
        self._data_reader = DataReader(**{'source': database})
        self._data_reader.Q_out = Queue(1)
        self._data_reader.start()
        # Create a DataTransformer
        self._data_transformer = DataTransformer(transform=transform,
                color_space=color_space, pack=True)
        self._data_transformer.Q_in = self._data_reader.Q_out
        self._data_transformer.Q_out = self.Q
        self._data_transformer.start()

        def cleanup():
            def terminate(process):
                process.terminate()
                process.join()

            terminate(self._data_transformer)
            terminate(self._data_reader)

        import atexit
        atexit.register(cleanup)

    def get(self):
        return self.Q.get()

    def next(self):
        return self.get()


class Dataset(object):
    """An abstract class representing a Dataset.

    Parameters
    ----------
    database : str
        The path of LMDB database.
    transform : lambda
        The transforms.
    color_space : str
        The color space.

    """

    def __init__(self, database, transform=None, color_space='RGB'):
        self.database = database
        self.transform = transform
        self.color_space = color_space

    def create_lmdb_stream(self):
        return _LMDBStream(self.database, self.transform, self.color_space)