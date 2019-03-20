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

from dragon.operators.misc import Run

from . import *


def LMDBData(**kwargs):
    """Prefetch Image data with `LMDB`_ database.

    Parameters
    ----------
    source : str
        The path of database.
    shuffle : bool, optional, default=False
        Whether to shuffle the data.
    node_step: bool
        Whether to split data for multiple parallel nodes.
    num_chunks : int, optional, default=2048
        The number of chunks to split.
    chunk_size : int, optional, default=-1
        The size(MB) of each chunk.
    mean_values : list, optional
        The mean value of each image channel.
    scale : float, optional, default=1.
        The scale performed after mean subtraction.
    padding : int, optional, default=0
        The zero-padding size.
    crop_size : int, optional, default=0
        The cropping size.
    mirror : bool, optional, default=False
        Whether to mirror(flip horizontally) images.
    color_augmentation : bool, optional, default=False
        Whether to use color distortion.1
    min_random_scale : float, optional, default=1.
        The min scale of the input images.
    max_random_scale : float, optional, default=1.
        The max scale of the input images.
    force_gray : bool, optional, default=False
        Set not to duplicate channel for gray.
    phase : {'TRAIN', 'TEST'}, optional
        The phase of this operator.
    batch_size : int, optional, default=128
        The size of a mini-batch.
    partition : bool, optional, default=False
        Whether to partition batch for parallelism.
    prefetch : int, optional, default=5
        The prefetch count.

    Returns
    -------
    sequence of Tensor
        The data and labels respectively.

    References
    ----------
    `DataBatch`_ - How to get a minibatch.

    `DataReader`_ - How to read data from `LMDB`_.

    `DataTransformer`_ - How to transform and augment data.

    `BlobFetcher`_ - How to form blobs.

    """
    arguments = ParseArgs(locals())
    arguments['module'] = 'dragon.operators.custom.minibatch'
    arguments['op'] = 'MiniBatchOp'
    return Run([], param_str=str(kwargs), num_outputs=2, **arguments)


@OpSchema.Inputs(1)
def ImageData(
    inputs, mean_values=None, std_values=None,
        dtype='float32', data_format='NCHW', **kwargs):
    """Process the images from 4D raw data.

    Note that we assume the data format of raw data is **NHWC**.

    Parameters
    ----------
    inputs : Tensor
        The input tensor, with type of **uint8** or **float32**.
    mean_values : sequence of float, optional
        The optional mean values to subtract.
    std_values : sequence of float, optional
        The optional std values to divide.
    dtype : {'float16', 'float32'}, optional
        The data type of output.
    data_format : {'NCHW', 'NHWC'}, optional
        The data format of output.

    Returns
    -------
    Tensor
        The output tensor.

    """
    arguments = ParseArgs(locals())

    if mean_values is not None:
        if len(mean_values) != 3:
            raise ValueError('The length of mean values should be 3.')
        arguments['mean_values'] = [float(v) for v in mean_values]

    if std_values is not None:
        if len(std_values) != 3:
            raise ValueError('The length of std values should be 3.')
        arguments['std_values'] = [float(v) for v in std_values]

    arguments['dtype'] = arguments['dtype'].lower()

    return Tensor.CreateOperator('ImageData', **arguments)