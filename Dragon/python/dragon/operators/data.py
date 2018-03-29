# ------------------------------------------------------------
# Copyright (c) 2017-preseent, SeetaTech, Co.,Ltd.
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


def ImageData(inputs, mean_values=None, std_values=None,
              dtype='FLOAT32', data_format='NCHW', **kwargs):
    """Process the images from 4D raw data.

    Note that we assume the data format of raw data is **NHWC**.

    Parameters
    ----------
    inputs : Tensor
        The input tensor, with type of **uint8** or **float32**.
    mean_values : list of float or None
        The optional mean values to subtract.
    std_values : list of float or None
        The optional std values to divide.
    dtype : str
        The type of output. ``FLOAT32`` or ``FLOAT16``.
    data_format : str
        The data format of output. ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArguments(locals())
    if mean_values is not None:
        if len(mean_values) != 3:
            raise ValueError('The length of mean values should be 3.')
        arguments['mean_values'] = [float(v) for v in mean_values]
    if std_values is not None:
        if len(std_values) != 3:
            raise ValueError('The length of std values should be 3.')
        arguments['std_values'] = [float(v) for v in std_values]

    return Tensor.CreateOperator(nout=1, op_type='ImageData', **arguments)