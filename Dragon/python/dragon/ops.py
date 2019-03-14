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

from .operators import initializer as init_ops
from .operators import vision as vision_ops
from .operators import loss as loss_ops
from .operators import data as data_ops
from .operators import activation as active_ops
from .operators import arithmetic as math_ops
from .operators import control_flow as control_flow_ops
from .operators import misc as misc_ops
from .operators import mpi as mpi_ops
from .operators import array as array_ops
from .operators import norm as norm_ops
from .operators import recurrent as recurrent_ops
from .operators import contrib as contrib_ops

# Data
LMDBData = data_ops.LMDBData
ImageData = data_ops.ImageData

# Initializer
Fill = init_ops.Fill
RandomUniform = init_ops.RandomUniform
RandomNormal = init_ops.RandomNormal
TruncatedNormal = init_ops.TruncatedNormal
GlorotUniform = init_ops.GlorotUniform
GlorotNormal = init_ops.GlorotNormal

# Vision
Conv2d = vision_ops.Conv2d
DepthwiseConv2d = vision_ops.DepthwiseConv2d
ConvTranspose2d = DeConv2d = Conv2dTranspose = vision_ops.ConvTranspose2d
Pool2d = vision_ops.Pool2d
ROIPool = vision_ops.ROIPool
ROIAlign = vision_ops.ROIAlign
LRN = vision_ops.LRN
NNResize = vision_ops.NNResize
BilinearResize = vision_ops.BilinearResize
BiasAdd = vision_ops.BiasAdd
DropBlock2d = vision_ops.DropBlock2d

# Recurrent
LSTMCell = recurrent_ops.LSTMCell
RNN = recurrent_ops.RNN
LSTM = recurrent_ops.LSTM
GRU = recurrent_ops.GRU

# Activation
Sigmoid = active_ops.Sigmoid
Tanh = active_ops.Tanh
Relu = active_ops.Relu
LRelu = active_ops.LRelu
PRelu = active_ops.PRelu
Elu = active_ops.Elu
SElu = active_ops.SElu
Softmax = active_ops.Softmax
Dropout = active_ops.Dropout

# Loss
NLLLoss = loss_ops.NLLLoss
SparseSoftmaxCrossEntropy = loss_ops.SparseSoftmaxCrossEntropy
SigmoidCrossEntropy = loss_ops.SigmoidCrossEntropy
SoftmaxCrossEntropy = loss_ops.SoftmaxCrossEntropy
SmoothL1Loss = loss_ops.SmoothL1Loss
L1Loss = loss_ops.L1Loss
L2Loss = loss_ops.L2Loss
SigmoidFocalLoss = loss_ops.SigmoidFocalLoss
SoftmaxFocalLoss = loss_ops.SoftmaxFocalLoss
CTCLoss = loss_ops.CTCLoss

# Arithmetic
Add = math_ops.Add
Sub = math_ops.Sub
Mul = math_ops.Mul
Div = math_ops.Div
Maximum = math_ops.Maximum
Minimum = math_ops.Minimum
Moments = math_ops.Moments
Clip = math_ops.Clip
Matmul = math_ops.Matmul
Pow = math_ops.Pow
Dot = math_ops.Dot
Log = math_ops.Log
Exp = math_ops.Exp
Square = math_ops.Square
Sqrt = math_ops.Sqrt
FullyConnected = math_ops.FullyConnected
Eltwise = math_ops.Eltwise
Affine = math_ops.Affine
GramMatrix = math_ops.GramMatrix
Accumulate = math_ops.Accumulate
MovingAverage = math_ops.MovingAverage

# Normalization
BatchNorm = norm_ops.BatchNorm
GroupNorm = norm_ops.GroupNorm
LayerNorm = norm_ops.LayerNorm
InstanceNorm = norm_ops.InstanceNorm
L2Norm = norm_ops.L2Norm

# NDArray
Gather = array_ops.Gather
Crop = array_ops.Crop
Reduce = array_ops.Reduce
Sum = array_ops.Sum
Mean = array_ops.Mean
Max = array_ops.Max
ArgMax = array_ops.ArgMax
Min = array_ops.Min
ArgMin = array_ops.ArgMin
Slice = array_ops.Slice
Stack = array_ops.Stack
Concat = array_ops.Concat
Transpose = array_ops.Transpose
Repeat = array_ops.Repeat
Tile = array_ops.Tile
Pad = array_ops.Pad
OneHot = array_ops.OneHot
Flatten = array_ops.Flatten
Reshape = array_ops.Reshape
ExpandDims = array_ops.ExpandDims
Squeeze = array_ops.Squeeze
Shape = array_ops.Shape
Arange = array_ops.Arange
Multinomial = array_ops.Multinomial

# Control Flow
Copy = control_flow_ops.Copy
Equal = control_flow_ops.Equal
Less = control_flow_ops.Less
LessEqual = control_flow_ops.LessEqual
Greater = control_flow_ops.Greater
GreaterEqual = control_flow_ops.GreaterEqual

# Misc
Cast = AsType = misc_ops.Cast
Run = misc_ops.Run
Template = misc_ops.Template
Accuracy = misc_ops.Accuracy
StopGradient = misc_ops.StopGradient

# MPI
MPIBroadcast = mpi_ops.MPIBroadcast
MPIGather = mpi_ops.MPIGather

# Contrib
Proposal = contrib_ops.Proposal # R-CNN