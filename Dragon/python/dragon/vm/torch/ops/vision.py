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

from dragon.vm.torch.ops.primitive import MakeContext
from dragon.vm.torch.ops.factory import get_module
from dragon.vm.torch.ops.modules.vision import Resize2d
from dragon.vm.torch.ops.modules.vision import RoIPool, RoIAlign


def _resize_2d(input, op_type, dsize, fx, fy):
    if dsize is None:
        if fx < 0 or fy < 0:
            raise ValueError('Set fx and fy if dsize is None.')
    else:
        if len(dsize) != 2:
            raise ValueError('The dsize should be a list with 2 elements.')
    if dsize is None and (fy == -1.0 or fx == -1.0):
        raise RuntimeError('The dsize, fx/fy should be specified either.')
    ctx = MakeContext(inputs=[input])
    key = 'torch.ops.{}/{}:{}/dsize:{}/fx:{}/fy:{}'.format(
        op_type.lower(), ctx[0], ctx[1], '2' if dsize else 'none', fx, fy)
    module = get_module(Resize2d, key, ctx,
        op_type=op_type, dsize=dsize, fx=fx, fy=fy)
    return module.forward(input, dsize)


def nn_resize(input, dsize, fx=-1.0, fy=-1.0):
    return _resize_2d(input, 'NNResize', dsize, fx, fy)


def bilinear_resize(input, dsize, fx=-1.0, fy=-1.0):
    return _resize_2d(input, 'BilinearResize', dsize, fx, fy)


def roi_pool(feature, rois, pooled_h, pooled_w, spatial_scale):
    ctx = MakeContext(inputs=[feature])
    key = 'torch.ops.roi_pool/{}:{}/pool_h:{}/pool_w:{}/spatial_scale:{}'.format(
        ctx[0], ctx[1], pooled_h, pooled_w, spatial_scale)
    module = get_module(RoIPool, key, ctx, pooled_h=pooled_h,
        pooled_w=pooled_w, spatial_scale=spatial_scale)
    return module.forward(feature, rois)


def roi_align(feature, rois, pooled_h, pooled_w,
              spatial_scale, sampling_ratio=2):
    ctx = MakeContext(inputs=[feature])
    key = 'torch.ops.roi_align/{}:{}/pool_h:{}/pool_w:{}/' \
          'spatial_scale:{}/sampling_ratio:{}'.format(
        ctx[0], ctx[1], pooled_h, pooled_w, spatial_scale, sampling_ratio)
    module = get_module(RoIAlign, key, ctx, pooled_h=pooled_h,
        pooled_w=pooled_w, spatial_scale=spatial_scale, sampling_ratio=sampling_ratio)
    return module.forward(feature, rois)