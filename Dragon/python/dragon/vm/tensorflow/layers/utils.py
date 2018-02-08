# --------------------------------------------------------
# TensorFlow @ Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

def convert_data_format(data_format, ndim):
    if data_format == 'channels_last':
        if ndim in (3, 4, 5):
            return 'NHWC'
        else:
            raise ValueError('Input rank not supported:', ndim)
    elif data_format == 'channels_first':
        if ndim in (3, 4, 5):
            return 'NCHW'
        else:
            raise ValueError('Input rank not supported:', ndim)
    else:
        raise ValueError('Invalid data_format:', data_format)


def normalize_tuple(value, n, name):
    if isinstance(value, int):
        return (value,) * n
    else:
        try:
            value_tuple = tuple(value)
        except TypeError:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        if len(value_tuple) != n:
            raise ValueError('The `' + name + '` argument must be a tuple of ' +
                             str(n) + ' integers. Received: ' + str(value))
        for single_value in value_tuple:
            try:
                int(single_value)
            except ValueError:
                raise ValueError('The `' + name + '` argument must be a tuple of ' +
                                 str(n) + ' integers. Received: ' + str(value) + ' '
                                                                                 'including element ' + str(single_value) + ' of type' +
                                 ' ' + str(type(single_value)))
        return value_tuple


def normalize_data_format(value):
    data_format = value.lower()
    if data_format not in {'channels_first', 'channels_last'}:
        raise ValueError('The `data_format` argument must be one of '
                         '"channels_first", "channels_last". Received: ' +
                         str(value))
    return data_format


def normalize_padding(value):
    padding = value.lower()
    if padding not in {'valid', 'same'}:
        raise ValueError('The `padding` argument must be one of "valid", "same". '
                         'Received: ' + str(padding))
    return padding
