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

from .operators import initializer as init
from .operators import vision
from .operators import loss
from .operators import data
from .operators import activation as act
from .operators import arithmetic as math
from .operators import control_flow
from .operators import misc as misc
from .operators import mpi
from .operators import ndarray
from .operators import norm
from .operators import recurrent
from .operators import contrib

# data
LMDBData = data.LMDBData
ImageData = data.ImageData

# init
Fill = init.Fill
RandomUniform = init.RandomUniform
RandomNormal = init.RandomNormal
TruncatedNormal = init.TruncatedNormal
GlorotUniform = init.GlorotUniform
GlorotNormal = init.GlorotNormal

# vision
Conv2d = vision.Conv2d
Conv2dTranspose = vision.Conv2dTranspose
Deconv2d = vision.Conv2dTranspose
Pool2d = vision.Pool2d
ROIPooling = vision.ROIPooling
ROIAlign = vision.ROIAlign
LRN = vision.LRN
NNResize = vision.NNResize
BilinearResize = vision.BilinearResize
BiasAdd = vision.BiasAdd
DenseConcat = vision.DenseConcat
DropBlock2d = vision.DropBlock2d

# recurrent
LSTMCell = recurrent.LSTMCell
RNN = recurrent.RNN
LSTM = recurrent.LSTM
GRU = recurrent.GRU

# activation
Sigmoid = act.Sigmoid
Tanh = act.Tanh
Relu = act.Relu
LRelu = act.LRelu
PRelu = act.PRelu
Elu = act.Elu
SElu = act.SElu
Softmax = act.Softmax
Dropout = act.Dropout

# loss
NLLLoss = loss.NLLLoss
SparseSoftmaxCrossEntropy = loss.SparseSoftmaxCrossEntropy
SigmoidCrossEntropy = loss.SigmoidCrossEntropy
SoftmaxCrossEntropy = loss.SoftmaxCrossEntropy
SmoothL1Loss = loss.SmoothL1Loss
L1Loss = loss.L1Loss
L2Loss = loss.L2Loss
SigmoidFocalLoss = loss.SigmoidFocalLoss
SoftmaxFocalLoss = loss.SoftmaxFocalLoss
CTCLoss = loss.CTCLoss

# arithmetic
Add = math.Add
Sub = math.Sub
Mul = math.Mul
Div = math.Div
Maximum = math.Maximum
Minimum = math.Minimum
Clip = math.Clip
Matmul = math.Matmul
Pow = math.Pow
Dot = math.Dot
Log = math.Log
Exp = math.Exp
Square = math.Square
Sqrt = math.Sqrt
InnerProduct = math.InnerProduct
Eltwise = math.Eltwise
Affine = math.Affine
GramMatrix = math.GramMatrix

# normalization
BatchNorm = norm.BatchNorm
BatchRenorm = norm.BatchRenorm
GroupNorm = norm.GroupNorm
FusedBatchNorm = norm.FusedBatchNorm
FusedGroupNorm = norm.FusedGroupNorm
InstanceNorm = norm.InstanceNorm
L2Norm = norm.L2Norm

# ndarray
Gather = ndarray.Gather
RandomPick = ndarray.RandomPick
Crop = ndarray.Crop
Reduce = ndarray.Reduce
Sum = ndarray.Sum
Mean = ndarray.Mean
Max = ndarray.Max
Argmax = ndarray.Argmax
Min = ndarray.Min
Argmin = ndarray.Argmin
Slice = ndarray.Slice
Stack = ndarray.Stack
Concat = ndarray.Concat
Transpose = ndarray.Transpose
Repeat = ndarray.Repeat
Tile = ndarray.Tile
Pad = ndarray.Pad
OneHot = ndarray.OneHot
Flatten = ndarray.Flatten
Reshape = ndarray.Reshape
ExpandDims = ndarray.ExpandDims
Squeeze = ndarray.Squeeze
Shape = ndarray.Shape
Arange = ndarray.Arange

# control flow
Copy = control_flow.Copy
Equal = control_flow.Equal

# misc
AsType = misc.AsType
Run = misc.Run
Template = misc.Template
Accuracy = misc.Accuracy
StopGradient = misc.StopGradient
MovingAverage = misc.MovingAverage

# mpi
MPIBroadcast = mpi.MPIBroadcast
MPIGather = mpi.MPIGather

# contrib
Proposal = contrib.Proposal # R-CNN