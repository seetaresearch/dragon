# --------------------------------------------------------
# Dragon
# Copyright(c) 2017 SeetaTech
# Written by Ting Pan
# --------------------------------------------------------

from .operators import initializer as init
from .operators import vision as vision
from .operators import loss as loss
from .operators import data as data
from .operators import activation as act
from .operators import arithmetic as math
from .operators import utils as utils
from .operators import cast as cast
from .operators import mpi as mpi
from .operators import common as common
from .operators import norm as norm
from .operators import recurrent as recurrent

# data
LMDBData = data.LMDBData
MemoryData = data.MemoryData

# init
Fill = init.Fill
RandomUniform = init.RandomalUniform
RandomNormal = init.RandomalNormal
TruncatedNormal = init.TruncatedNormal
GlorotUniform = init.GlorotUniform
GlorotNormal = init.GlorotNormal

# vision
Conv2D = vision.Conv2D
Deconv2D = vision.Deconv2D
Pool2D = vision.Pool2D
ROIPooling = vision.ROIPooling
ROIAlign = vision.ROIAlign
LRN = vision.LRN
NNResize = vision.NNResize
BiasAdd = vision.BiasAdd
DenseConcat = vision.DenseConcat

# recurrent
LSTMUnit = recurrent.LSTMUnit

# activation
Sigmoid = act.Sigmoid
Tanh = act.Tanh
Relu = act.Relu
LRelu = act.LRelu
Softmax = act.Softmax
Dropout = act.Dropout

# loss
SparseSoftmaxCrossEntropy = loss.SparseSoftmaxCrossEntropy
SigmoidCrossEntropy = loss.SigmoidCrossEntropy
SoftmaxCrossEntropy = loss.SoftmaxCrossEntropy
SmoothL1Loss = loss.SmoothL1Loss
L1Loss = loss.L1Loss
L2Loss = loss.L2Loss
SparseSoftmaxFocalLoss = loss.SparseSoftmaxFocalLoss

# arithmetic
Add = math.Add
Sub = math.Sub
Mul = math.Mul
Div = math.Div
Clip = math.Clip
Matmul = math.Matmul
Pow = math.Pow
Dot = math.Dot
Log = math.Log
Exp = math.Exp
Square = math.Square
InnerProduct = math.InnerProduct
Eltwise = math.Eltwise
Scale = math.Scale
Argmax = math.Argmax
GramMatrix = math.GramMatrix

# normalization
BatchNorm = norm.BatchNorm
BatchRenorm = norm.BatchRenorm
BN = norm.BN
InstanceNorm = norm.InstanceNorm
L2Norm = norm.L2Norm

# common
At = common.At
Crop = common.Crop
Reduce = common.Reduce
Sum = common.Sum
Mean = common.Mean
Slice = common.Slice
Concat = common.Concat
Transpose = common.Transpose
Tile = common.Tile
Flatten = common.Flatten
Reshape = common.Reshape
ExpandDims = common.ExpandDims
Shape = common.Shape

# utils
Run = utils.Run
Template = utils.Template
Accuracy = utils.Accuracy
StopGradient = utils.StopGradient
OneHot = utils.OneHot
MovingAverage = utils.MovingAverage
Copy = utils.Copy
Equal = utils.Equal
Proposal = utils.Proposal

# cast
FloatToHalf = cast.FloatToHalf

# mpi
MPIBroadcast = mpi.MPIBroadcast
MPIGather = mpi.MPIGather