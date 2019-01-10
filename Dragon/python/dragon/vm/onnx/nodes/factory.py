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

from collections import Iterable

from dragon.vm.onnx.nodes.common import CommonONNXExporter, UnsupportedONNXExporter
from dragon.vm.onnx.nodes.activation import SoftmaxONNXExporter
from dragon.vm.onnx.nodes.arithmetic import GemmONNXExporter, AffineONNXExporter
from dragon.vm.onnx.nodes.norm import BatchNormONNXExporter, L2NormONNXExporter

from dragon.vm.onnx.nodes.misc import (
    ImageDataONNXExporter,
    AsTypeONNXExporter,
    PythonONNXExporter,
)

from dragon.vm.onnx.nodes.ndarray import (
    ReshapeONNXExporter,
    ConcatONNXExporter,
    FlattenONNXExporter,
    TransposeONNXExporter,
    CropONNXExporter,
    ArgReduceONNXExporter,
)

from dragon.vm.onnx.nodes.vision import (
    ConvNdONNXExporter,
    PoolNdONNXExporter,
    ResizeNdONNXExporter,
    ROIPoolONNXExporter,
    ROIAlignONNXExporter,
)

from dragon.vm.onnx.nodes.contrib import ProposalONNXExporter


_special_exporters = {
    'Conv2d': ConvNdONNXExporter,
    'ConvTranspose2d': ConvNdONNXExporter,
    'Pool2d': PoolNdONNXExporter,
    'FullyConnected': GemmONNXExporter,
    'Softmax': SoftmaxONNXExporter,
    'BatchNorm': UnsupportedONNXExporter,
    'FusedBatchNorm': BatchNormONNXExporter,
    'NNResize': ResizeNdONNXExporter,
    'BilinearResize': ResizeNdONNXExporter,
    'Reshape': ReshapeONNXExporter,
    'Concat': ConcatONNXExporter,
    'Flatten': FlattenONNXExporter,
    'Proposal': ProposalONNXExporter,
    'ROIPool': ROIPoolONNXExporter,
    'ROIAlign': ROIAlignONNXExporter,
    'ImageData': ImageDataONNXExporter,
    'AsType': AsTypeONNXExporter,
    'Transpose': TransposeONNXExporter,
    'Template': PythonONNXExporter,
    'Run': PythonONNXExporter,
    'L2Norm': L2NormONNXExporter,
    'Affine': AffineONNXExporter,
    'Crop': CropONNXExporter,
    'ArgReduce': ArgReduceONNXExporter,
}


def get_nodes_def(op_def, shape_dict):
    if op_def.type in _special_exporters:
        nodes, const_tensors = \
            _special_exporters[op_def.type](op_def, shape_dict)
    else:
        nodes, const_tensors = \
            CommonONNXExporter(op_def, shape_dict)

    if not isinstance(nodes, Iterable):
        return [nodes], const_tensors