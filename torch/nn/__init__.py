# ------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
#
# Licensed under the BSD 2-Clause License.
# You should have received a copy of the BSD 2-Clause License
# along with the software. If not, See,
#
#    <https://opensource.org/licenses/BSD-2-Clause>
#
# ------------------------------------------------------------
"""The List of nn components."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.torch.nn import init
from dragon.vm.torch.nn.modules.activation import ELU
from dragon.vm.torch.nn.modules.activation import GumbelSoftmax
from dragon.vm.torch.nn.modules.activation import LeakyReLU
from dragon.vm.torch.nn.modules.activation import LogSoftmax
from dragon.vm.torch.nn.modules.activation import PReLU
from dragon.vm.torch.nn.modules.activation import ReLU
from dragon.vm.torch.nn.modules.activation import ReLU6
from dragon.vm.torch.nn.modules.activation import SELU
from dragon.vm.torch.nn.modules.activation import Sigmoid
from dragon.vm.torch.nn.modules.activation import Softmax
from dragon.vm.torch.nn.modules.activation import Tanh
from dragon.vm.torch.nn.modules.affine import Affine
from dragon.vm.torch.nn.modules.batchnorm import BatchNorm1d
from dragon.vm.torch.nn.modules.batchnorm import BatchNorm2d
from dragon.vm.torch.nn.modules.batchnorm import BatchNorm3d
from dragon.vm.torch.nn.modules.batchnorm import SyncBatchNorm
from dragon.vm.torch.nn.modules.container import Container
from dragon.vm.torch.nn.modules.container import ModuleList
from dragon.vm.torch.nn.modules.container import Sequential
from dragon.vm.torch.nn.modules.conv import Conv2d
from dragon.vm.torch.nn.modules.conv import ConvTranspose2d
from dragon.vm.torch.nn.modules.conv import DepthwiseConv2d
from dragon.vm.torch.nn.modules.dropout import DropBlock2d
from dragon.vm.torch.nn.modules.dropout import Dropout
from dragon.vm.torch.nn.modules.dropout import DropPath
from dragon.vm.torch.nn.modules.linear import Linear
from dragon.vm.torch.nn.modules.loss import CTCLoss
from dragon.vm.torch.nn.modules.loss import BCEWithLogitsLoss
from dragon.vm.torch.nn.modules.loss import CrossEntropyLoss
from dragon.vm.torch.nn.modules.loss import L1Loss
from dragon.vm.torch.nn.modules.loss import MSELoss
from dragon.vm.torch.nn.modules.loss import NLLLoss
from dragon.vm.torch.nn.modules.loss import SigmoidFocalLoss
from dragon.vm.torch.nn.modules.loss import SmoothL1Loss
from dragon.vm.torch.nn.modules.loss import SCEWithLogitsLoss
from dragon.vm.torch.nn.modules.module import Module
from dragon.vm.torch.nn.modules.normalization import GroupNorm
from dragon.vm.torch.nn.modules.normalization import LocalResponseNorm
from dragon.vm.torch.nn.modules.padding import ConstantPad1d
from dragon.vm.torch.nn.modules.padding import ConstantPad2d
from dragon.vm.torch.nn.modules.padding import ConstantPad3d
from dragon.vm.torch.nn.modules.padding import ReflectionPad1d
from dragon.vm.torch.nn.modules.padding import ReflectionPad2d
from dragon.vm.torch.nn.modules.padding import ReflectionPad3d
from dragon.vm.torch.nn.modules.padding import ReplicationPad1d
from dragon.vm.torch.nn.modules.padding import ReplicationPad2d
from dragon.vm.torch.nn.modules.padding import ReplicationPad3d
from dragon.vm.torch.nn.modules.padding import ZeroPad2d
from dragon.vm.torch.nn.modules.pooling import MaxPool2d
from dragon.vm.torch.nn.modules.pooling import AvgPool2d
from dragon.vm.torch.nn.modules.pooling import MaxPool2d
from dragon.vm.torch.nn.modules.rnn import GRU
from dragon.vm.torch.nn.modules.rnn import LSTM
from dragon.vm.torch.nn.modules.rnn import LSTMCell
from dragon.vm.torch.nn.modules.rnn import RNN
from dragon.vm.torch.nn.modules.rnn import RNNBase
from dragon.vm.torch.nn.modules.rnn import RNNCellBase
from dragon.vm.torch.nn.modules.upsampling import Upsample
from dragon.vm.torch.nn.modules.upsampling import UpsamplingBilinear2d
from dragon.vm.torch.nn.modules.upsampling import UpsamplingNearest2d
from dragon.vm.torch.nn.parameter import Parameter
