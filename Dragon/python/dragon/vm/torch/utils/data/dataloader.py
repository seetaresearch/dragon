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

from .data_batch import DataBatch as _Batch


class _DataLoaderIter(object):
    def __init__(self, loader):
        self.loader = loader

    def __len__(self):
        return len(self.loader.batch.Q3.qsize())

    def __next__(self):
        return self.loader.batch.get()

    next = __next__  # Python 2 compatibility

    def __iter__(self):
        return self


class DataLoader(object):
    def __init__(
        self,
        dataset,
        batch_size=1,
        shuffle=False,
        num_chunks=2048,
        phase='TRAIN',
    ):
        """A MPI-Aware DataLoader. Forked from ``dragon.utils.vision``.

        Parameters
        ----------
        dataset : torch.utils.data.dataset.Dataset
            The dataset.
        batch_size : int, optional, default=1
            The batch size.
        shuffle : boolean, optional, default=False
            Whether to shuffle the data.
        num_chunks : int, optional, default=2048
            The number of chunks to split.
        phase : {'TRAIN', 'TEST'}, optional
            The optional running phase.

        """
        self.dataset = dataset
        self.batch_size = batch_size
        n_transformers = 1
        if dataset.transform and \
            hasattr(dataset.transform, 'n_transformers'):
                n_transformers = dataset.transform.n_transformers
        self.batch = _Batch(**{
            'source': dataset.database,
            'shuffle': shuffle,
            'num_chunks': num_chunks,
            'phase': phase,
            'batch_size': batch_size,
            'transform': dataset.transform,
            'color_space': dataset.color_space,
            'num_transformers': n_transformers,
        })

    def __iter__(self):
        return _DataLoaderIter(self)

    def __len__(self):
        return self.batch.Q3.qsize()

    def next(self):
        return self.batch.get()

    def get(self):
        return self.batch.get()