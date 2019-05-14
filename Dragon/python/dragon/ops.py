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

"""A simple collector for implemented operators."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from .operators import initializer as _init_ops
from .operators import vision as _vision_ops
from .operators import loss as _loss_ops
from .operators import data as _data_ops
from .operators import activation as _active_ops
from .operators import arithmetic as _math_ops
from .operators import control_flow as _control_flow_ops
from .operators import misc as _misc_ops
from .operators import mpi as _mpi_ops
from .operators import array as _array_ops
from .operators import norm as _norm_ops
from .operators import recurrent as _recurrent_ops
from .operators import contrib as _contrib_ops

# Data
LMDBData = _data_ops.LMDBData
ImageData = _data_ops.ImageData

# Initializer
Fill = _init_ops.Fill
RandomUniform = _init_ops.RandomUniform
RandomNormal = _init_ops.RandomNormal
TruncatedNormal = _init_ops.TruncatedNormal
GlorotUniform = _init_ops.GlorotUniform
GlorotNormal = _init_ops.GlorotNormal

# Vision
Conv2d = _vision_ops.Conv2d
DepthwiseConv2d = _vision_ops.DepthwiseConv2d
ConvTranspose2d = DeConv2d = Conv2dTranspose = _vision_ops.ConvTranspose2d
Pool2d = _vision_ops.Pool2d
ROIPool = _vision_ops.ROIPool
ROIAlign = _vision_ops.ROIAlign
LRN = _vision_ops.LRN
NNResize = _vision_ops.NNResize
BilinearResize = _vision_ops.BilinearResize
BiasAdd = _vision_ops.BiasAdd
DropBlock2d = _vision_ops.DropBlock2d

# Recurrent
LSTMCell = _recurrent_ops.LSTMCell
RNN = _recurrent_ops.RNN
LSTM = _recurrent_ops.LSTM
GRU = _recurrent_ops.GRU

# Activation
Sigmoid = _active_ops.Sigmoid
Tanh = _active_ops.Tanh
Relu = _active_ops.Relu
LRelu = _active_ops.LRelu
PRelu = _active_ops.PRelu
Elu = _active_ops.Elu
SElu = _active_ops.SElu
Softmax = _active_ops.Softmax
Dropout = _active_ops.Dropout
DropPath = _active_ops.DropPath

# Loss
NLLLoss = _loss_ops.NLLLoss
SparseSoftmaxCrossEntropy = _loss_ops.SparseSoftmaxCrossEntropy
SigmoidCrossEntropy = _loss_ops.SigmoidCrossEntropy
SoftmaxCrossEntropy = _loss_ops.SoftmaxCrossEntropy
SmoothL1Loss = _loss_ops.SmoothL1Loss
L1Loss = _loss_ops.L1Loss
L2Loss = _loss_ops.L2Loss
SigmoidFocalLoss = _loss_ops.SigmoidFocalLoss
SoftmaxFocalLoss = _loss_ops.SoftmaxFocalLoss
CTCLoss = _loss_ops.CTCLoss

# Arithmetic
Add = _math_ops.Add
Sub = _math_ops.Sub
Mul = _math_ops.Mul
Div = _math_ops.Div
Maximum = _math_ops.Maximum
Minimum = _math_ops.Minimum
Moments = _math_ops.Moments
Clip = _math_ops.Clip
Matmul = _math_ops.Matmul
Pow = _math_ops.Pow
Dot = _math_ops.Dot
Log = _math_ops.Log
Exp = _math_ops.Exp
Square = _math_ops.Square
Sqrt = _math_ops.Sqrt
FullyConnected = _math_ops.FullyConnected
Eltwise = _math_ops.Eltwise
Affine = _math_ops.Affine
GramMatrix = _math_ops.GramMatrix
Accumulate = _math_ops.Accumulate
MovingAverage = _math_ops.MovingAverage

# Normalization
BatchNorm = _norm_ops.BatchNorm
GroupNorm = _norm_ops.GroupNorm
LayerNorm = _norm_ops.LayerNorm
InstanceNorm = _norm_ops.InstanceNorm
L2Norm = _norm_ops.L2Norm

# NDArray
Crop = _array_ops.Crop
Reduce = _array_ops.Reduce
Sum = _array_ops.Sum
Mean = _array_ops.Mean
Max = _array_ops.Max
ArgMax = _array_ops.ArgMax
Min = _array_ops.Min
ArgMin = _array_ops.ArgMin
Slice = _array_ops.Slice
Stack = _array_ops.Stack
Concat = _array_ops.Concat
Transpose = _array_ops.Transpose
Repeat = _array_ops.Repeat
Tile = _array_ops.Tile
Pad = _array_ops.Pad
IndexSelect = _array_ops.IndexSelect
OneHot = _array_ops.OneHot
Flatten = _array_ops.Flatten
Reshape = _array_ops.Reshape
ExpandDims = _array_ops.ExpandDims
Squeeze = _array_ops.Squeeze
Shape = _array_ops.Shape
Arange = _array_ops.Arange
Multinomial = _array_ops.Multinomial

# Control Flow
Copy = _control_flow_ops.Copy
Assign = _control_flow_ops.Assign
MaskedAssign = _control_flow_ops.MaskedAssign
Equal = _control_flow_ops.Equal
Less = _control_flow_ops.Less
LessEqual = _control_flow_ops.LessEqual
Greater = _control_flow_ops.Greater
GreaterEqual = _control_flow_ops.GreaterEqual

# Misc
Cast = AsType = _misc_ops.Cast
Run = _misc_ops.Run
Template = _misc_ops.Template
Accuracy = _misc_ops.Accuracy
StopGradient = _misc_ops.StopGradient

# MPI
MPIBroadcast = _mpi_ops.MPIBroadcast
MPIGather = _mpi_ops.MPIGather

# Contrib
Proposal = _contrib_ops.Proposal # R-CNN