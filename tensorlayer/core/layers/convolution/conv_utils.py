# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
# Copyright (c) 2016-2018, The TensorLayer contributors.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import functools

from dragon.core.util import nest
from dragon.core.util import six


def convert_data_format(data_format):
    """
    Convert data_data_format.

    Args:
        data_format: (str): write your description
    """
    if data_format == 'channels_last':
        return 'NHWC'
    elif data_format == 'channels_first':
        return 'NCHW'
    else:
        raise ValueError('Invalid data_format: ' + data_format)


def normalize_data_format(value):
    """
    Normalize data format.

    Args:
        value: (str): write your description
    """
    if value is None:
        value = 'channels_first'
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError(
            'Excepted <data_format> as one of '
            '"channels_first", '
            '"channels_last".'
            ' Received: ' + str(value))
    return data_format


def normalize_spatial_args(name, values, num_spatial_dims):
    """
    Normalize arguments.

    Args:
        name: (str): write your description
        values: (str): write your description
        num_spatial_dims: (int): write your description
    """
    if name in ('ksize', 'strides', 'dilations'):
        if values is None:
            return [1] * num_spatial_dims
        else:
            values = nest.flatten(values)
            if len(values) == 1:
                return [values[0]] * num_spatial_dims
            elif len(values) != num_spatial_dims:
                defaults = [1] * num_spatial_dims
                defaults[:num_spatial_dims] = values
                return defaults
            return values
    elif name == 'padding':
        if isinstance(values, six.string_types):
            padding, pads = values.upper(), 0
        else:
            padding_tuple = nest.flatten(values)
            padding = 'VALID'
            if len(padding_tuple) == 1:
                pads = padding_tuple[0]
            elif len(padding_tuple) == num_spatial_dims:
                pads = padding_tuple
            elif len(padding_tuple) == (num_spatial_dims * 2):
                pads_l, pads_r = [], []
                for i in range(num_spatial_dims):
                    pads_l.append(padding_tuple[i * 2])
                    pads_r.append(padding_tuple[i * 2 + 1])
                pads = pads_l + pads_r
            else:
                raise ValueError(
                    'Except 1, {} or {} values if <padding> set as explict pads.'
                    .format(num_spatial_dims, num_spatial_dims * 2)
                )
        return padding, pads


# Aliases
normalize_1d_args = functools.partial(normalize_spatial_args, num_spatial_dims=1)
normalize_2d_args = functools.partial(normalize_spatial_args, num_spatial_dims=2)
normalize_3d_args = functools.partial(normalize_spatial_args, num_spatial_dims=3)
