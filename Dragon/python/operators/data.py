# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

import numpy as np
from dragon.core.tensor import Tensor
from dragon.operators.utils import Run

def LMDBData(**kwargs):
    """
    :param kwargs:                   a dict of imagenet data param
    :param --> mean_value:           a list of mean values for channles [B-G-R]
    :param --> source:               a str of the images root directory
    :param --> imageset:             a str of text file contains image name / label
    :param --> prefetch:             a int of the prefetching size
    :param --> batch_size:           a int of the batch size
    :param --> force_gray            a bool of whether to use only 1 channel
    :param --> shuffle               a bool of whether to use shuffle
    :param --> crop_size             a int
    :param --> mirror                a bool
    :param --> color_augmentation    a bool
    :param --> min_random_scale      a float, defualt is 1.0
    :param --> max_random_scale      a float, default is 1.0
    :param --> scale                 a float of the coeff to scale
    :return:                         2 Tensors of data and label
    """

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)

    kwargs['module'] =  'dragon.vm.caffe.io.data_layer'
    kwargs['op'] = 'DataLayer'

    return Run([], param_str=str(kwargs), nout=2, **kwargs)


def MemoryData(inputs, dtype=np.float32, **kwargs):

    args = locals(); kwargs = args['kwargs']
    del args['kwargs']; kwargs = dict(args, **kwargs)
    if dtype is np.float32: kwargs['dtype'] = 1
    elif dtype is np.float16: kwargs['dtype'] = 12
    else: raise TypeError('unsupported data type')

    return Tensor.CreateOperator(nout=1, op_type='MemoryData', **kwargs)