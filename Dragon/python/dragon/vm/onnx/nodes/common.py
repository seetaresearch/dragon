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

from onnx.helper import make_node


_renamed_operators = {
    'Conv2d': 'Conv',
    'ConvTranspose2d': 'ConvTranspose',
    'Pool2d': 'MaxPool',
    'FullyConnected': 'Gemm',
    'FusedBatchNorm': 'BatchNormalization',
    'NNResize': 'Upsample',
    'BilinearResize': 'Upsample',
    'ROIPool': 'MaxRoiPool',
    'L2Norm': 'LpNormalization',
}


def CommonONNXExporter(op_def, shape_dict):
    return make_node(
        op_type=_renamed_operators[op_def.type] \
            if op_def.type in _renamed_operators
                else op_def.type,
        inputs=op_def.input,
        outputs=op_def.output,
        name=op_def.name if op_def.name != '' else None
    ), None


def UnsupportedONNXExporter(op_def, shape_dict):
    raise ValueError('{} is not supported by ONNX.'.format(op_def.type))