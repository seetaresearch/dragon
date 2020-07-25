# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# Codes are based on:
#
#     <https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/keras/utils/conv_utils.py>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.util import nest
from dragon.core.util import six


def convert_data_format(data_format, ndim):
    if data_format == 'channels_last':
        if ndim == 3:
            return 'NWC'
        elif ndim == 4:
            return 'NHWC'
        elif ndim == 5:
            return 'NDHWC'
        else:
            raise ValueError('Input rank not supported: ' + ndim)
    elif data_format == 'channels_first':
        if ndim == 3:
            return 'NCW'
        elif ndim == 4:
            return 'NCHW'
        elif ndim == 5:
            return 'NCDHW'
        else:
            raise ValueError('Input rank not supported: ' + ndim)
    else:
        raise ValueError('Invalid data_format: ' + data_format)


def deconv_output_length(
    input_length,
    filter_size,
    padding,
    output_padding=None,
    stride=0,
    dilation=1,
):
    assert padding in {'same', 'valid', 'full'}
    if input_length is None:
        return None
    filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if output_padding is None:
        if padding == 'full':
            length = input_length * stride - (stride + filter_size - 2)
        elif padding == 'same':
            length = input_length * stride
        else:
            length = input_length * stride + max(filter_size - stride, 0)
    else:
        if padding == 'same':
            pad = filter_size // 2
        elif padding == 'full':
            pad = filter_size - 1
        else:
            pad = 0
        length = ((input_length - 1) * stride + filter_size - 2 * pad + output_padding)
    return length


def normalize_data_format(value):
    if value is None:
        value = 'channels_last'
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError(
            'Excepted <data_format> as one of '
            '"channels_first", '
            '"channels_last".'
            ' Received: ' + str(value))
    return data_format


def normalize_padding(value):
    """Return the padding descriptor."""
    if isinstance(value, six.string_types):
        value = value.lower()
        if value not in {'valid', 'same'}:
            raise ValueError(
                'Excepted <padding> in "valid", "same".\n'
                'Received: ' + str(value))
    return value


def normalize_tuple(value, rank):
    """Repeat the value according to the rank."""
    value = nest.flatten(value)
    if len(value) > rank:
        return (value[i] for i in range(rank))
    else:
        return tuple([value[i] for i in range(len(value))] +
                     [value[-1] for _ in range(len(value), rank)])
