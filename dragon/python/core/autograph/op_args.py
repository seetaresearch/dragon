# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#     <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""Operator arguments."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from dragon.core.framework import device_spec
from dragon.core.util import registry

_REGISTERED_OP_ARGS = registry.Registry('OpArgs')
get = _REGISTERED_OP_ARGS.try_get
register = _REGISTERED_OP_ARGS.register


@register('Accuracy')
def accuracy_args(**kwargs):
    return {
        'no_grad': True,
        'axis': kwargs.get('axis', -1),
        'top_k': kwargs.get('top_k', 1),
        'ignore_index': kwargs.get('ignore_index', None),
    }


@register(['ArgMax', 'ArgMin'])
def arg_reduce_args(**kwargs):
    return {
        'no_grad': True,
        'axis': kwargs.get('axis', None),
        'keepdims': kwargs.get('keepdims', True),
    }


@register('Assign')
def assign_args(**kwargs):
    return {
        'no_grad': True,
        'starts_desc': 'int64',
        'sizes_desc': 'int64',
    }


@register('Axpby')
def axpby_args(**kwargs):
    return {
        'alpha': kwargs.get('alpha', 1.0),
        'beta': kwargs.get('beta', 0.0),
    }


@register('BatchNorm')
def batch_norm_args(**kwargs):
    return {
        'axis': kwargs.get('axis', -1),
        'epsilon': kwargs.get('epsilon', 1e-5),
        'use_stats': kwargs.get('use_stats', 0),
        'momentum_desc': 'float32',
    }


@register('BiasAdd')
def bias_add_args(**kwargs):
    return {'data_format': kwargs.get('data_format', 'NCHW')}


@register('Cast')
def cast_args(**kwargs):
    return {'dtype': kwargs.get('dtype', 'float32')}


@register('Affine')
def affine_args(**kwargs):
    return {'axes': kwargs.get('axes', None)}


@register('ChannelNorm')
def channel_norm_args(**kwargs):
    return {
        'axis': kwargs.get('axis', -1),
        'mean': kwargs.get('mean', None),
        'std': kwargs.get('std', None),
        'dtype': kwargs.get('dtype', 'float32'),
        'perm_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
    }


@register('ChannelShuffle')
def channel_shuffle_args(**kwargs):
    return {'axis': kwargs.get('axis', 0), 'group': kwargs.get('group', 1)}


@register('Clip')
def clip_args(**kwargs):
    return {'low': kwargs.get('low', None), 'high': kwargs.get('high', None)}


@register('Collective')
def collective_args(**kwargs):
    return {
        'operation': kwargs.get('operation', ''),
        'reduction': kwargs.get('reduction', 'MEAN'),
        'root': kwargs.get('root', 0),
        'comm': kwargs.get('comm', 0),
        'group': kwargs.get('group', 0),
        'backend': kwargs.get('backend', 'MPI'),
        'ranks': kwargs.get('ranks', None),
    }


@register('Concat')
def concat_args(**kwargs):
    return {'axis': kwargs.get('axis', 0)}


@register(['Conv', 'DepthwiseConv', 'Im2Col'])
def conv_args(**kwargs):
    return {
        'kernel_shape': kwargs.get('kernel_shape', 1),
        'strides': kwargs.get('strides', 1),
        'pads': kwargs.get('pads', 0),
        'dilations': kwargs.get('dilations', 1),
        'group': kwargs.get('group', None),
        'padding': kwargs.get('padding', 'VALID'),
        'data_format': kwargs.get('data_format', 'NCHW'),
    }


@register(['Col2Im', 'ConvTranspose'])
def conv_transpose_args(**kwargs):
    return {**conv_args(**kwargs), **{
        'output_padding': kwargs.get('output_padding', None),
        'output_shape': kwargs.get('output_shape', None),
    }}


@register('CumSum')
def cum_reduce_args(**kwargs):
    return {
        'axis': kwargs.get('axis', None),
        'exclusive': kwargs.get('exclusive', False),
        'reverse': kwargs.get('reverse', False),
    }


@register(['DepthToSpace', 'SpaceToDepth'])
def depth_space_args(**kwargs):
    return {
        'block_size': kwargs.get('block_size', '2'),
        'mode': kwargs.get('mode', 'DCR'),
        'data_format': kwargs.get('data_format', 'NCHW'),
    }


@register(['Dropout', 'DropPath'])
def dropout_args(**kwargs):
    return {'ratio_desc': 'float32'}


@register('DropBlock')
def drop_block_args(**kwargs):
    return {
        'ratio_desc': 'float32',
        'block_size': kwargs.get('block_size', 1),
        'data_format': kwargs.get('data_format', 'NCHW'),
    }


@register('Elu')
def elu_args(**kwargs):
    return {'alpha': kwargs.get('alpha', 1.0)}


@register('Expand')
def expand_args(**kwargs):
    return {'dims_desc': 'int64'}


@register('Eye')
def eye_args(**kwargs):
    return {
        'no_grad': True,
        'dtype': kwargs.get('dtype', 'float32'),
        'dims_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
        'k': kwargs.get('k', 0),
    }


@register('Fill')
def fill_args(**kwargs):
    return {
        'no_grad': True,
        'dtype': kwargs.get('dtype', 'float32'),
        'dims_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
        'value': kwargs.get('value', 0.0),
    }


@register('Flatten')
def flatten_args(**kwargs):
    return {
        'axis': kwargs.get('axis', 0),
        'end_axis': kwargs.get('end_axis', -1),
    }


@register('SigmoidFocalLoss')
def focal_loss_args(**kwargs):
    return {
        'axis': kwargs.get('axis', -1),
        'alpha': kwargs.get('alpha', 0.25),
        'gamma': kwargs.get('gamma', 2.0),
        'start_index': kwargs.get('start_index', None),
        'reduction': kwargs.get('reduction', 'valid'),
    }


@register('Gather')
def gather_args(**kwargs):
    return {
        'axis': kwargs.get('axis', 0),
        'end_axis': kwargs.get('end_axis', kwargs.get('axis', 0)),
    }


@register('Gelu')
def gelu_args(**kwargs):
    return {'approximate': kwargs.get('approximate', False)}


@register('Gemm')
def gemm_args(**kwargs):
    return {
        'alpha': kwargs.get('alpha', 1.0),
        'beta': kwargs.get('beta', 1.0),
        'transA': kwargs.get('transA', False),
        'transB': kwargs.get('transB', False),
    }


@register('GlorotNormal')
def glorot_normal_args(**kwargs):
    return {
        'no_grad': True,
        'scale': kwargs.get('scale', 2.0),
        'mode': kwargs.get('mode', 'fan_in').lower(),
        'dtype': kwargs.get('dtype', 'float32'),
        'dims_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
    }


@register('GlorotUniform')
def glorot_uniform_args(**kwargs):
    return {
        'no_grad': True,
        'scale': kwargs.get('scale', 3.0),
        'mode': kwargs.get('mode', 'fan_in').lower(),
        'dtype': kwargs.get('dtype', 'float32'),
        'dims_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
    }


@register('GroupNorm')
def group_norm_args(**kwargs):
    return {
        'axis': kwargs.get('axis', -1),
        'group': kwargs.get('group', 32),
        'epsilon': kwargs.get('epsilon', 1e-5),
    }


@register('HardSigmoid')
def hard_sigmoid_args(**kwargs):
    return {
        'alpha': kwargs.get('alpha', 0.2),
        'beta': kwargs.get('beta', 0.5),
    }


@register('LayerNorm')
def layer_norm_args(**kwargs):
    return {
        'axis': kwargs.get('axis', -1),
        'epsilon': kwargs.get('epsilon', 1e-5),
    }


@register('LinSpace')
def linspace_args(**kwargs):
    return {
        'axis': kwargs.get('axis', 0),
        'dtype': kwargs.get('dtype', 'int64'),
        'dims_desc': 'int64',
        'start_desc': 'float64',
        'stop_desc': 'float64',
    }


@register('LRN')
def local_response_norm_args(**kwargs):
    return {
        'size': kwargs.get('size', 5),
        'alpha': kwargs.get('alpha', 0.0001),
        'beta': kwargs.get('beta', 0.75),
        'bias': kwargs.get('bias', 1.0),
    }


@register(['L1Loss', 'L2Loss', 'SigmoidCrossEntropyLoss'])
def loss_args(**kwargs):
    return {'reduction': kwargs.get('reduction', 'MEAN')}


@register('LpNorm')
def lp_norm_args(**kwargs):
    return {
        'p': kwargs.get('p', 2),
        'axis': kwargs.get('axis', -1),
        'end_axis': kwargs.get('end_axis', kwargs.get('axis', -1)),
        'epsilon': kwargs.get('epsilon', 1e-12),
        'reduction': kwargs.get('reduction', 'SUM'),
    }


@register('Multinomial')
def multinomial_args(**kwargs):
    return {'sample_size': kwargs.get('sample_size', 1)}


@register(['NLLLoss', 'SoftmaxCrossEntropyLoss'])
def nll_loss_args(**kwargs):
    return {
        'axis': kwargs.get('axis', -1),
        'ignore_index': kwargs.get('ignore_index', None),
        'reduction': kwargs.get('reduction', 'valid')
    }


@register('OneHot')
def one_hot_args(**kwargs):
    return {
        'depth': kwargs.get('depth', 1),
        'on_value': kwargs.get('on_value', 1.0),
        'off_value': kwargs.get('off_value', 0.0),
    }


@register('Pad')
def pad_args(**kwargs):
    return {
        'value': kwargs.get('value', 0.),
        'mode': kwargs.get('mode', 'CONSTANT'),
        'pads_desc': 'int64',
    }


@register('Permutation')
def permutation_args(**kwargs):
    return {'dtype': kwargs.get('dtype', 'int64'), 'limit_desc': 'int64'}


@register('Pool')
def pool_args(**kwargs):
    return {
        'kernel_shape': kwargs.get('kernel_shape', 1),
        'strides': kwargs.get('strides', 1),
        'pads': kwargs.get('pads', 0),
        'padding': kwargs.get('padding', 'VALID'),
        'ceil_mode': kwargs.get('ceil_mode', False),
        'mode': kwargs.get('mode', 'MAX'),
        'global_pool': kwargs.get('global_pool', False),
        'data_format': kwargs.get('data_format', 'NCHW'),
    }


@register('PRelu')
def prelu_args(**kwargs):
    return {'data_format': kwargs.get('data_format', 'NCHW')}


@register('RandomNormal')
def random_normal_args(**kwargs):
    return {
        'no_grad': True,
        'mean': kwargs.get('mean', 0.0),
        'std': kwargs.get('std', 1.0),
        'dtype': kwargs.get('dtype', 'float32'),
        'dims_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
    }


@register('RandomUniform')
def random_uniform_args(**kwargs):
    return {
        'no_grad': True,
        'low': kwargs.get('low', 0.0),
        'high': kwargs.get('high', 1.0),
        'dtype': kwargs.get('dtype', 'float32'),
        'dims_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
    }


@register('Range')
def range_args(**kwargs):
    return {'dtype': kwargs.get('dtype', 'int64'), 'slice_desc': 'float64'}


@register(['ReduceMax',
           'ReduceMin',
           'ReduceMean',
           'ReduceSum',
           'ReduceVar',
           'ReduceL1',
           'ReduceL2',
           'Moments'])
def reduce_args(**kwargs):
    return {
        'axes': kwargs.get('axes', None),
        'keepdims': kwargs.get('keepdims', True),
    }


@register('Relu')
def relu_args(**kwargs):
    return {
        'alpha': kwargs.get('alpha', 0.0),
        'max_value': kwargs.get('max_value', None),
    }


@register('Repeat')
def repeat_args(**kwargs):
    return {
        'axis': kwargs.get('axis', 0),
        'repeats_desc': 'int64',
    }


@register('Reshape')
def reshape_args(**kwargs):
    return {'dims_desc': 'int64' if kwargs.get('ndim', 0) else None}


@register('Resize')
def resize_args(**kwargs):
    return {
        'mode': kwargs.get('mode', 'NEAREST'),
        'align_corners': kwargs.get('align_corners', False),
        'data_format': kwargs.get('data_format', 'NCHW'),
        'sizes_desc': 'int64' if kwargs.get('num_sizes', 0) > 0 else None,
        'scales_desc': 'float32' if kwargs.get('num_scales', 0) > 0 else None,
    }


@register('Reverse')
def reverse_args(**kwargs):
    return {'axes': kwargs.get('axes', None)}


@register('Recurrent')
def rnn_args(**kwargs):
    return {
        'num_layers': kwargs.get('num_layers', 1),
        'bidirectional': kwargs.get('bidirectional', 0),
        'input_size': kwargs.get('input_size', 0),
        'hidden_size': kwargs.get('hidden_size', 0),
        'dropout': kwargs.get('dropout', 0.0),
        'phase': kwargs.get('phase', 'TEST'),
        'rnn_mode': kwargs.get('rnn_mode', 'rnn_tanh'),
    }


@register('RNNParamSet')
def rnn_param_args(**kwargs):
    return {
        'no_grad': True,
        'num_layers': kwargs.get('num_layers', 1),
        'bidirectional': kwargs.get('bidirectional', 0),
        'input_size': kwargs.get('input_size', 0),
        'hidden_size': kwargs.get('hidden_size', 0),
        'layer_id': kwargs.get('layer_id', 0),
        'param_id': kwargs.get('param_id', 0),
        'param_type': kwargs.get('param_type', 'matrix'),
        'rnn_mode': kwargs.get('rnn_mode', 'rnn_tanh'),
    }


@register('RoiAlign')
def roi_align_args(**kwargs):
    return {
        'pooled_h': kwargs.get('pooled_h', 7),
        'pooled_w': kwargs.get('pooled_w', 7),
        'spatial_scale': kwargs.get('spatial_scale', 1.0),
        'sampling_ratio': kwargs.get('sampling_ratio', 0),
        'aligned': kwargs.get('aligned', False),
    }


@register('RoiPool')
def roi_pool_args(**kwargs):
    return {
        'pooled_h': kwargs.get('pooled_h', 7),
        'pooled_w': kwargs.get('pooled_w', 7),
        'spatial_scale': kwargs.get('spatial_scale', 1.0),
    }


@register('Roll')
def roll_args(**kwargs):
    return {
        'axes': kwargs.get('axes', None),
        'shifts_desc': 'int64',
    }


@register(['ScatterElements', 'ScatterAdd', 'GatherElements'])
def scatter_gather_elements_args(**kwargs):
    return {'axis': kwargs.get('axis', 0)}


@register('Selu')
def selu_args(**kwargs):
    return {
        'alpha': kwargs.get('alpha', 1.67326),
        'gamma': kwargs.get('gamma', 1.0507),
    }


@register('Shape')
def shape_args(**kwargs):
    return {'device': device_spec.DeviceSpec('cpu')}


@register('Slice')
def slice_args(**kwargs):
    return {'starts_desc': 'int64', 'sizes_desc': 'int64'}


@register('SmoothL1Loss')
def smooth_l1_loss_args(**kwargs):
    return {
        'beta': kwargs.get('beta', 1.0),
        'reduction': kwargs.get('reduction', 'valid'),
    }


@register(['Softmax', 'LogSoftmax'])
def softmax_args(**kwargs):
    return {'axis': kwargs.get('axis', -1)}


@register('Sort')
def sort_args(**kwargs):
    return {
        'axis': kwargs.get('axis', -1),
        'descending': kwargs.get('descending', False),
    }


@register('Split')
def split_args(**kwargs):
    return {
        'axis': kwargs.get('axis', 0),
        'copy': kwargs.get('copy', True),
        'keepdims': kwargs.get('keepdims', True),
        'split_desc': 'int64' if kwargs.get('num_splits', 0) > 0 else None,
    }


@register('Squeeze')
def squeeze_args(**kwargs):
    return {'axes': kwargs.get('axes', None)}


@register('Stack')
def stack_args(**kwargs):
    return {'axis': kwargs.get('axis', 0)}


@register('SyncBatchNorm')
def sync_batch_norm_args(**kwargs):
    return {**batch_norm_args(**kwargs), **{
        'comm': kwargs.get('comm', 0),
        'group': kwargs.get('group', 0),
        'backend': kwargs.get('backend', 'MPI'),
        'ranks': kwargs.get('ranks', None),
    }}


@register('Tile')
def tile_args(**kwargs):
    return {'repeats_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None}


@register('TopK')
def top_k_args(**kwargs):
    return {
        'k': kwargs.get('k', 1),
        'axis': kwargs.get('axis', -1),
        'largest': kwargs.get('largest', True),
        'sorted': kwargs.get('sorted', True),
    }


@register('Transpose')
def transpose_args(**kwargs):
    return {'perm_desc': 'int64' if kwargs.get('ndim', 0) else None}


@register('Trilu')
def triangular_args(**kwargs):
    return {'k': kwargs.get('k', 0), 'upper': kwargs.get('upper', False)}


@register('TruncatedNormal')
def trucated_normal_args(**kwargs):
    return {
        'no_grad': True,
        'mean': kwargs.get('mean', 0.0),
        'std': kwargs.get('std', 1.0),
        'low': kwargs.get('low', -2.0),
        'high': kwargs.get('high', 2.0),
        'dtype': kwargs.get('dtype', 'float32'),
        'dims_desc': 'int64' if kwargs.get('ndim', 0) > 0 else None,
    }


@register('Unique')
def unique_args(**kwargs):
    return {
        'return_inverse': kwargs.get('return_inverse', False),
        'return_counts': kwargs.get('return_counts', False),
    }


@register('Unsqueeze')
def unsqueeze_args(**kwargs):
    return {'axes': kwargs.get('axes', [0])}


@register(['Adam',
           'AdamW',
           'RMSprop',
           'MomentumSGD',
           'NesterovSGD',
           'LARS'])
def update_args(**kwargs):
    return {'no_grad': True, 'weight_decay': kwargs.get('weight_decay', None)}
