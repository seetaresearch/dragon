# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
from six.moves import range as xrange

from . import *

def Conv2d(inputs, num_output, kernel_size,
           stride=1, pad=0, dilation=1, group=1,
           padding='VALID', data_format='NCHW', **kwargs):
    """2D Convolution.

    The number of inputs vary from ``2`` to ``3`` (Without or With ``bias``).

    The spatial output dimension of convolution can be computed as follows:

    |conv_output_dim|

    Set ``padding`` to  **VALID** will use the value of ``pad``.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, weights, bias].
    num_output : int
        The output channels of convolution.
    kernel_size : int or list
        The kernel size(s) of convolution.
    stride : int or list
        The stride(s) of convolution. Default is ``1``.
    pad : int or list
        The zero padding size(s) of convolution. Default is ``0``.
    dilation : int or list
        The dilation multiple(s) of convolution. Default is ``1``.
    group : int
        The group size of convolution. Default is ``1``.
    padding : str
        The padding algorithm. ``VALID`` or ``SAME``.
    data_format : str
        The data format. ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> input = Tensor().Variable()
    >>> weights = Tensor().Normal(std=0.001)
    >>> biases = Tensor().Constant(value=0)
    >>> conv1 = Conv2d([input, weights, biases], num_output=64, kernel_size=3)

    >>> weights = Tensor().Gaussian(std=0.001)
    >>> conv2 = Conv2d([conv1, weights], num_output=128, kernel_size=3, stride=1)

    """
    CheckInputs(inputs, 2, 3)
    arguments = ParseArguments(locals())

    if padding not in ('VALID', 'SAME'):
        raise ValueError('Unsupported padding algorithm: {}'.format(padding))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if not isinstance(arguments['kernel_size'], list):
        arguments['kernel_size'] = [arguments['kernel_size']]
    if not isinstance(arguments['stride'], list):
        arguments['stride'] = [arguments['stride']]
    if not isinstance(arguments['pad'], list):
        arguments['pad'] = [arguments['pad']]
    if not isinstance(arguments['dilation'], list):
        arguments['dilation'] = [arguments['dilation']]

    output = Tensor.CreateOperator(nout=1, op_type='Conv2d', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]
        channel_axis = 1 if data_format == 'NCHW' else -1
        spatial_axis = 2 if data_format == 'NCHW' else 1
        output.shape[channel_axis] = num_output
        for i in xrange(2):
            k = arguments['kernel_size'][i] if i < len(arguments['kernel_size']) \
                                            else arguments['kernel_size'][-1]
            s = arguments['stride'][i]      if i < len(arguments['stride']) \
                                            else arguments['stride'][-1]
            p = arguments['pad'][i]         if i < len(arguments['pad']) \
                                            else arguments['pad'][-1]
            d = arguments['dilation'][i]    if i < len(arguments['dilation']) \
                                            else arguments['dilation'][-1]
            dk = d * (k - 1) + 1
            dp = 2 * p
            if padding == 'SAME':
                input_size = output.shape[i + spatial_axis]
                output_size = (input_size + s - 1) / float(s)
                dp = int(max(0, (output_size - 1) * s + k - input_size))
            output.shape[i + spatial_axis] = int(output.shape[i + spatial_axis] + dp - dk / s) + 1

    return output


def Conv2dTranspose(inputs, num_output, kernel_size,
                    stride=1, pad=0, dilation=1, group=1, output_shape=None,
                    padding='VALID', data_format='NCHW', **kwargs):
    """2D Deconvolution.

    The number of inputs vary from ``2`` to ``3`` (Without or With ``bias``).

    The spatial output dimension of deconvolution can be computed as follows:

    |deconv_output_dim|

    Set ``padding`` to  **VALID** will use the value of ``pad``.

    Provide ``output_shape`` if using **SAME** padding.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs of deconvolution, represent [input, weights, bias].
    num_output : int
        The output channels of deconvolution.
    kernel_size : int or list
        The kernel size(s) of deconvolution.
    stride : int or list
        The stride(s) of deconvolution. Default is ``1``.
    pad : int or list
        The zero padding size(s) of deconvolution. Default is ``0``.
    dilation : int or list
        The dilation multiple(s) of deconvolution. Default is ``1``.
    group : int
        The group size of deconvolution. Default is ``1``.
    output_shape : list of int or None
        The deterministic output shape for **SAME** padding.
    padding : str
        The padding algorithm. ``VALID`` or ``SAME``.
    data_format : str
        The data format. ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor.

    Examples
    --------
    >>> input = Tensor().Variable()
    >>> weights = Tensor().Normal(std=0.001)
    >>> biases = Tensor().Constant(value=0)
    >>> deconv1 = Conv2dTranspose([input, weights, biases], num_output=64, kernel_size=3)

    >>> weights = Tensor().Gaussian(std=0.001)
    >>> deconv2 = Conv2dTranspose([deconv1, weights], num_output=128, kernel_size=3, stride=1)

    """
    CheckInputs(inputs, 2, 3)
    arguments = ParseArguments(locals())

    if padding not in ('VALID', 'SAME'):
        raise ValueError('Unsupported padding algorithm: {}'.format(padding))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    arguments['output_shape'] = None
    if output_shape is not None:
        if not isinstance(output_shape, list):
            raise TypeError('The output shape should be a list.')
        if isinstance(output_shape[0], Tensor):
            arguments['dynamic_dsize'] = []
            arguments['extra_inputs'] = list(output_shape)
            for dim in output_shape:
                arguments['dynamic_dsize'].append(dim)
        else:
            arguments['static_dsize'] = []
            for dim in output_shape:
                arguments['static_dsize'].append(int(dim))

    if not isinstance(arguments['kernel_size'], list):
        arguments['kernel_size'] = [arguments['kernel_size']]
    if not isinstance(arguments['stride'], list):
        arguments['stride'] = [arguments['stride']]
    if not isinstance(arguments['pad'], list):
        arguments['pad'] = [arguments['pad']]
    if not isinstance(arguments['dilation'], list):
        arguments['dilation'] = [arguments['dilation']]

    output =  Tensor.CreateOperator(nout=1, op_type='Conv2dTranspose', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]
        channel_axis = 1 if data_format == 'NCHW' else -1
        spatial_axis = 2 if data_format == 'NCHW' else 1
        output.shape[channel_axis] = num_output
        for i in xrange(2):
            k = arguments['kernel_size'][i] if i < len(arguments['kernel_size']) \
                else arguments['kernel_size'][-1]
            s = arguments['stride'][i] if i < len(arguments['stride']) \
                else arguments['stride'][-1]
            p = arguments['pad'][i] if i < len(arguments['pad']) \
                else arguments['pad'][-1]
            d = arguments['dilation'][i] if i < len(arguments['dilation']) \
                else arguments['dilation'][-1]
            dk = d * (k - 1) + 1
            dp = 2 * p
            input_size = output.shape[i + spatial_axis]
            if padding != 'SAME':
                output.shape[i + spatial_axis] = s * (input_size - 1) + dk - dp
            else:
                if output_shape is None:
                    raise ValueError('The output shape must be specified if using SAME padding algorithm.')
                if 'dynamic_dsize' in arguments:
                    output.shape = None
                    return output
                output.shape[i + spatial_axis] = output_shape[i + spatial_axis]

    return output


def Pool2d(inputs, kernel_size, stride, pad=0, padding='VALID',
           mode='MAX', data_format='NCHW', global_pooling=False, **kwargs):
    """2D Pooling, MAX or AVG.

    The spatial output dimension of pooling can be computed as follows:

    |pooling_output_dim|

    Set ``padding`` to  **VALID** will use the value of ``pad``.

    If use ``global_pooling``, the stride and pad will be set to ``1`` and ``0`` automatically.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    kernel_size : int or list
        The kernel size(s) of pooling.
    stride : int or list
        The stride(s) of of pooling,
    pad : int or list
        The zero padding size(s) of pooling. Default is ``0``.
    padding : str
        The padding algorithm. ``VALID`` or ``SAME``.
    mode : str
        The mode, ``MAX`` or ``AVG``.
    data_format : str
        The data format, ``NCHW`` or ``NHWC``.
    global_pooling : boolean
        Whether to use global pooling.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    if mode not in ('MAX', 'AVG'):
        raise ValueError('Unsupported lrn mode: {}'.format(mode))
    if padding not in ('VALID', 'SAME'):
        raise ValueError('Unsupported padding algorithm: {}'.format(padding))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if not isinstance(arguments['kernel_size'], list):
        arguments['kernel_size'] = [arguments['kernel_size']]
    if not isinstance(arguments['stride'], list):
        arguments['stride'] = [arguments['stride']]
    if not isinstance(arguments['pad'], list):
        arguments['pad'] = [arguments['pad']]

    output = Tensor.CreateOperator(nout=1, op_type='Pooling2d', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]
        spatial_axis = 2 if data_format == 'NCHW' else 1
        for i in xrange(2):
            k = arguments['kernel_size'][i] if i < len(arguments['kernel_size']) \
                                            else arguments['kernel_size'][-1]
            s = arguments['stride'][i]      if i < len(arguments['stride']) \
                                            else arguments['stride'][-1]
            p = arguments['pad'][i]         if i < len(arguments['pad']) \
                                            else arguments['pad'][-1]
            if not global_pooling:
                if padding != 'SAME':
                    input_size = output.shape[i + spatial_axis]
                    output_size = int(math.ceil((output.shape[i + spatial_axis] + 2 * p - k) / s) + 1)
                    if ((output_size - 1) * s >= input_size + p):
                        output_size = output_size - 1
                    output.shape[i + spatial_axis] = output_size
                else:
                    output.shape[i + spatial_axis] = \
                        int((output.shape[i + spatial_axis] + s - 1) / s)
            else:
                output.shape[i + spatial_axis] = 1

    return output


def ROIPooling(inputs, pool_h, pool_w, spatial_scale, **kwargs):
    """Max ROIPooling, introduced by `[Girshick, 2015] <https://arxiv.org/abs/1504.08083>`_.

    The first dimension of input must be ``1``.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent input and RoIs respectively.
    pool_h : int
        The height of pooled tensor.
    pool_w : int
        The width of pooled tensor.
    spatial_scale : float
        The ``inverse`` of total down-sampling multiples on input tensor.

    Returns
    -------
    Tensor
        The batch of pooled RoI regions.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())
    return Tensor.CreateOperator(nout=1, op_type='ROIPooling', **arguments)


def ROIAlign(inputs, pool_h=0, pool_w=0, spatial_scale=1.0, **kwargs):
    """Max ROIAlign, introduced by `[He et.al, 2017] <https://arxiv.org/abs/1703.06870>`_.

    The first dimension of input must be ``1``.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent input and RoIs respectively.
    pool_h : int
        The height of pooled tensor.
    pool_w : int
        The width of pooled tensor.
    spatial_scale : float
        The ``inverse`` of total down-sampling multiples on input tensor.

    Returns
    -------
    Tensor
        The batch of pooled RoI regions.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())
    return Tensor.CreateOperator(nout=1, op_type='ROIAlign', **arguments)


def LRN(inputs, local_size=5, alpha=0.0001, beta=0.75, k=2.0,
        mode='ACROSS_CHANNELS', data_format='NCHW', **kwargs):
    """Local Response Normalization, introduced by `[Krizhevsky et.al, 2012] <http://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks>`_.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    local_size : int
        The local size of LRN.
    alpha : float
        The alpha of LRN.
    beta : float
        The beta of LRN.
    k : float
        The k of LRN.
    mode : str
        The mode, ``ACROSS_CHANNELS`` or ``WITHIN_CHANNEL``.
    data_format : str
        The data format. ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    if mode not in ('ACROSS_CHANNELS', 'WITHIN_CHANNEL'):
        raise ValueError('Unsupported lrn mode: {}'.format(mode))
    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    output = Tensor.CreateOperator(nout=1, op_type='LRN', **arguments)

    if inputs.shape is not None:
        output.shape = inputs.shape[:]

    return output


def NNResize(inputs, dsize, fy=-1.0, fx=-1.0, data_format='NCHW', **kwargs):
    """Resize the image with Nearest-Neighbor method.

    Set ``dsize`` to None if you want to use ``fy`` and ``fx``.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    dsize : tuple, list, Tensor or None
        The output size, formats as (h, w).
    fy : float
        The scale factor based on src height. Default is ``-1.0`` (Discarded).
    fx : float
        The scale factor based on src width. Default is ``-1.0`` (Discarded).
    data_format : str
        The data_format. ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if arguments['dsize'] is not None:
        if isinstance(arguments['dsize'][0], Tensor):
            arguments['dynamic_dsize'] = [arguments['dsize'][0].name,
                                          arguments['dsize'][1].name]
            arguments['extra_inputs'] = list(arguments['dsize'])
        else:
            arguments['static_size'] = arguments['dsize']
        del arguments['dsize']

    if dsize is None and (fy == -1.0 or fx == -1.0):
        raise RuntimeError('The dsize or fy/fx should be specified either.')

    output =  Tensor.CreateOperator(nout=1, op_type='NNResize', **arguments)

    if inputs.shape is not None:
        if len(inputs.shape) != 4:
            raise ValueError('The inputs should be a 4d Tensor.')
        if 'dynamic_dsize' not in arguments:
            output.shape = inputs.shape[:]
            spatial_axis = 2 if data_format == 'NCHW' else 1
            for i in xrange(2):
                output_dim = output.shape[spatial_axis + i]
                if 'static_size' in arguments:
                    output_dim = dsize[i]
                else:
                    output_dim = int(float(output_dim) * ([fy, fx])[i])
                output.shape[spatial_axis + i] = output_dim

    return output


def BilinearResize(inputs, dsize, fy=-1.0, fx=-1.0, data_format='NCHW', **kwargs):
    """Resize the image with Bi-linear method.

    Set ``dsize`` to None if you want to use ``fy`` and ``fx``.

    Parameters
    ----------
    inputs : Tensor
        The input tensor.
    dsize : tuple, list, Tensor or None
        The output size, formats as (h, w).
    fy : float
        The scale factor based on src height. Default is ``-1.0`` (Discarded).
    fx : float
        The scale factor based on src width. Default is ``-1.0`` (Discarded).
    data_format : str
        The data_format. ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The output tensor.

    """
    CheckInputs(inputs, 1)
    arguments = ParseArguments(locals())

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    if arguments['dsize'] is not None:
        if isinstance(arguments['dsize'][0], Tensor):
            arguments['dynamic_dsize'] = [arguments['dsize'][0].name,
                                          arguments['dsize'][1].name]
            arguments['extra_inputs'] = list(arguments['dsize'])
        else:
            arguments['static_size'] = arguments['dsize']
        del arguments['dsize']

    if dsize is None and (fy == -1.0 or fx == -1.0):
        raise RuntimeError('The dsize or fy/fx should be specified either.')

    output =  Tensor.CreateOperator(nout=1, op_type='BilinearResize', **arguments)

    if inputs.shape is not None:
        if len(inputs.shape) != 4:
            raise ValueError('The inputs should be a 4d Tensor.')
        if 'dynamic_dsize' not in arguments:
            output.shape = inputs.shape[:]
            spatial_axis = 2 if data_format == 'NCHW' else 1
            for i in xrange(2):
                output_dim = output.shape[spatial_axis + i]
                if 'static_size' in arguments:
                    output_dim = dsize[i]
                else:
                    output_dim = int(float(output_dim) * ([fy, fx])[i])
                output.shape[spatial_axis + i] = output_dim

    return output


def BiasAdd(inputs, data_format='NCHW', **kwargs):
    """Add the bias across channels to a ``NCHW`` or ``NHWC`` input.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent [input, bias].
    data_format : str
        The data format, ``NCHW`` or ``NHWC``.

    Returns
    -------
    Tensor
        The bias-added tensor.

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())

    if data_format not in ('NCHW', 'NHWC'):
        raise ValueError('Unsupported data format: {}'.format(data_format))

    output =  Tensor.CreateOperator(nout=1, op_type='BiasAdd', **arguments)

    if inputs[0].shape is not None:
        output.shape = inputs[0].shape[:]

    return output


def DenseConcat(inputs, growth_rate=0, axis=1, **kwargs):
    """Memory-efficient concatenation for DenseNet `[Huang et.al, 2017] <http://arxiv.org/abs/1608.06993>`_.

    This operator is forked from ``Concat``.

    The memory optimization requires the following settings:

    1. Set the ``growth_rate``, the value must larger than ``0``.

    2. Set the ``mirror_stage`` to True.

    Parameters
    ----------
    inputs : list of Tensor
        The inputs, represent A(old) and B(new) respectively.
    growth_rate : int
        The growth rate. Default is ``0`` (Without Memory Optimization).
    axis : int
        The axis to concatenate.
    mirror_stage : boolean(optional)
        Whether to share input A for output C. Default is ``False``.

    Returns
    -------
    Tensor
        The concatenated tensor, represents C.

    Examples
    --------
    >>> A = Tensor().Variable()
    >>> B = Tensor().Variable()
    >>> C = DenseConcat([A, B], axis=1) # normal concatenation

    >>> import dragon.memonger as opt
    >>> C = opt.Drop(DenseConcat, [A, B], axis=1) #  memory-efficient concatenation

    >>> C = DenseConcat([A, B], axis=1, mirror_stage=True) #  memory-efficient concatenation, equivalent

    """
    CheckInputs(inputs, 2)
    arguments = ParseArguments(locals())
    arguments['num_input'] = len(inputs)

    output = Tensor.CreateOperator(nout=1, op_type='DenseConcat', **arguments)

    if all(input.shape is not None for input in inputs):
        if all(input.shape[axis] is not None for input in inputs):
            output.shape = inputs[0].shape[:]
            for i in xrange(1, len(inputs)):
                output.shape[axis] += inputs[i].shape[axis]

    return output