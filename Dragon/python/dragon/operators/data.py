# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
from dragon.operators.misc import Run

from . import *

def LMDBData(**kwargs):
    """Prefetch Image data with `LMDB`_ database.

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
    mean_values : list
        The mean value of each image channel.
    scale : float
        The scale performed after mean subtraction. Default is ``1.0``.
    padding : int
        The zero-padding size. Default is ``0``.
    crop_size : int
        The crop size. Default is ``0`` (Disabled).
    mirror : boolean
        Whether to mirror(flip horizontally) images. Default is ``False``.
    color_augmentation : boolean
        Whether to use color distortion. Default is ``False``.
    min_random_scale : float
        The min scale of the input images. Default is ``1.0``.
    max_random_scale : float
        The max scale of the input images. Default is ``1.0``.
    force_gray : boolean
        Set not to duplicate channel for gray. Default is ``False``.
    phase : str
        The phase of this operator, ``TRAIN`` or ``TEST``.
    batch_size : int
        The size of a mini-batch.
    partition : boolean
        Whether to partition batch for parallelism. Default is ``False``.
    prefetch : int
        The prefetch count. Default is ``5``.

    Returns
    -------
    list of Tensor.
        Two tensors, representing data and labels respectively.

    References
    ----------
    `DataBatch`_ - How to get a minibatch.

    `DataReader`_ - How to read data from `LMDB`_.

    `DataTransformer`_ - How to transform and augment data.

    `BlobFetcher`_ - How to form blobs.

    """
    arguments = ParseArguments(locals())
    arguments['module'] = 'dragon.operators.custom.minibatch'
    arguments['op'] = 'MiniBatchOp'

    return Run([], param_str=str(kwargs), nout=2, **arguments)


def MemoryData(inputs, dtype=np.float32, **kwargs):
    """Perform ``NHWC <-> NCHW``, ``Mean Subtraction`` and ``Type Converting``.

    Parameters
    ----------
    inputs : Tensor
        The input tensor, with type of uint8 or float32.
    dtype : np.float32 or np.float16
        The dtype of output tensor.

    Returns
    -------
    Tensor
        The post-processing Tensor.

    """
    arguments = ParseArguments(locals())
    if dtype is np.float32: arguments['dtype'] = 1
    elif dtype is np.float16: arguments['dtype'] = 12
    else: raise TypeError('Unsupported data type.')

    return Tensor.CreateOperator(nout=1, op_type='MemoryData', **arguments)