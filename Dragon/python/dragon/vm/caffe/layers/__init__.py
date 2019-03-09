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

"""Import all the implemented caffe layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Data Layers
from .data import (
    DataLayer,
    MemoryDataLayer,
)

# Vision Layers
from .vision import (
    ConvolutionLayer,
    DepthwiseConvolutionLayer,
    DeconvolutionLayer,
    PoolingLayer,
    LRNLayer,
    ROIPoolingLayer,
    ROIAlignLayer,
    NNResizeLayer,
    BilinearResizeLayer,
    DropBlockLayer,
)

# Neuron Layers
from .neuron import (
    ReLULayer,
    PReLULayer,
    ELULayer,
    SELULayer,
    DropoutLayer,
    SigmoidLayer,
    TanHLayer,
    PowerLayer,
)

# Loss Layers
from .loss import (
    SoftmaxWithLossLayer,
    SigmoidCrossEntropyLossLayer,
    L2LossLayer,
    SmoothL1LossLayer,
    SigmoidWithFocalLossLayer,
    SoftmaxWithFocalLossLayer,
)

# MPI Layers
from .mpi import (
    MPIBroadcastLayer,
    MPIGatherLayer,
)

# Common Layers
from .common import (
    InnerProductLayer,
    AccuracyLayer,
    BatchNormLayer,
    GroupNormLayer,
    BNLayer,
    GNLayer,
    ConcatLayer,
    CropLayer,
    PythonLayer,
    AddLayer,
    ReshapeLayer,
    EltwiseLayer,
    ScaleLayer,
    SoftmaxLayer,
    ArgMaxLayer,
    PermuteLayer,
    FlattenLayer,
    GatherLayer,
    ConcatLayer,
    NormalizeLayer,
    InstanceNormLayer,
    TileLayer,
    ReductionLayer,
    ExpandDimsLayer,
    StopGradientLayer,
    ProposalLayer,
    CastLayer,
)