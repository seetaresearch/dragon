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

from dragon.vm.torch.utils.data.io.data_batch import DataBatch as _DataBatch


class _DataLoaderIter(object):
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader.batch.Q_level_3.qsize())

    def __next__(self):
        return self.loader.batch.get()

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self


class DataLoader(object):
    def __init__(self, dataset, batch_size=1, shuffle=False,
                 partition=False, multiple_nodes=False, instance_chunk=False):
        """A MPI-Aware DataLoader. Forked from ``dragon.io``.

        Parameters
        ----------
        dataset : vm.torch.utils.data.dataset.Dataset
            The dataset.
        batch_size : int
            The batch size. Divided by n mpi-nodes if ``partition`` is True.
        instance_chunk : boolean
            Whether to limit each chunk with at most 1 instance.
        shuffle : boolean
            Whether to shuffle the data.
        partition : boolean
            Whether to partition batch. Default is ``False``.
        multiple_nodes: boolean
            Whether to split data for multiple parallel nodes.
        instance_chunk : boolean
            Whether to limit each chunk with at most 1 instance.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        n_transformers = 1
        if dataset.transform and \
            hasattr(dataset.transform, 'n_transformers'):
                n_transformers = dataset.transform.n_transformers
        self.batch = _DataBatch(**{
            'source': dataset.database,
            'multiple_nodes': multiple_nodes,
            'shuffle': shuffle,
            'instance_chunk': instance_chunk,
            'batch_size': batch_size,
            'partition': partition,
            'transform': dataset.transform,
            'color_space': dataset.color_space,
            'num_transformers': n_transformers,
        })

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return self.batch.Q_level_3.qsize()

    def next(self):
        return self.batch.get()

    def get(self):
        return self.batch.get()