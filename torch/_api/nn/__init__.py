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
"""NN module."""

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

# Modules
from dragon.vm.torch._api.nn import functional
from dragon.vm.torch._api.nn import init

# Classes
from dragon.vm.torch.core.nn.modules.activation import ELU
from dragon.vm.torch.core.nn.modules.activation import GELU
from dragon.vm.torch.core.nn.modules.activation import GumbelSoftmax
from dragon.vm.torch.core.nn.modules.activation import Hardsigmoid
from dragon.vm.torch.core.nn.modules.activation import Hardswish
from dragon.vm.torch.core.nn.modules.activation import LeakyReLU
from dragon.vm.torch.core.nn.modules.activation import LogSoftmax
from dragon.vm.torch.core.nn.modules.activation import MultiheadAttention
from dragon.vm.torch.core.nn.modules.activation import PReLU
from dragon.vm.torch.core.nn.modules.activation import ReLU
from dragon.vm.torch.core.nn.modules.activation import ReLU6
from dragon.vm.torch.core.nn.modules.activation import SELU
from dragon.vm.torch.core.nn.modules.activation import Sigmoid
from dragon.vm.torch.core.nn.modules.activation import SiLU
from dragon.vm.torch.core.nn.modules.activation import Softmax
from dragon.vm.torch.core.nn.modules.activation import Tanh
from dragon.vm.torch.core.nn.modules.batchnorm import BatchNorm1d
from dragon.vm.torch.core.nn.modules.batchnorm import BatchNorm2d
from dragon.vm.torch.core.nn.modules.batchnorm import BatchNorm3d
from dragon.vm.torch.core.nn.modules.batchnorm import SyncBatchNorm
from dragon.vm.torch.core.nn.modules.channelshuffle import ChannelShuffle
from dragon.vm.torch.core.nn.modules.container import Container
from dragon.vm.torch.core.nn.modules.container import ModuleList
from dragon.vm.torch.core.nn.modules.container import Sequential
from dragon.vm.torch.core.nn.modules.conv import Conv1d
from dragon.vm.torch.core.nn.modules.conv import Conv2d
from dragon.vm.torch.core.nn.modules.conv import Conv3d
from dragon.vm.torch.core.nn.modules.conv import ConvTranspose1d
from dragon.vm.torch.core.nn.modules.conv import ConvTranspose2d
from dragon.vm.torch.core.nn.modules.conv import ConvTranspose3d
from dragon.vm.torch.core.nn.modules.conv import DepthwiseConv2d
from dragon.vm.torch.core.nn.modules.distance import CosineSimilarity
from dragon.vm.torch.core.nn.modules.dropout import DropBlock2d
from dragon.vm.torch.core.nn.modules.dropout import Dropout
from dragon.vm.torch.core.nn.modules.dropout import DropPath
from dragon.vm.torch.core.nn.modules.flatten import Flatten
from dragon.vm.torch.core.nn.modules.fold import Unfold
from dragon.vm.torch.core.nn.modules.linear import Affine
from dragon.vm.torch.core.nn.modules.linear import Identity
from dragon.vm.torch.core.nn.modules.linear import Linear
from dragon.vm.torch.core.nn.modules.loss import CTCLoss
from dragon.vm.torch.core.nn.modules.loss import BCEWithLogitsLoss
from dragon.vm.torch.core.nn.modules.loss import CrossEntropyLoss
from dragon.vm.torch.core.nn.modules.loss import KLDivLoss
from dragon.vm.torch.core.nn.modules.loss import L1Loss
from dragon.vm.torch.core.nn.modules.loss import MSELoss
from dragon.vm.torch.core.nn.modules.loss import NLLLoss
from dragon.vm.torch.core.nn.modules.loss import SigmoidFocalLoss
from dragon.vm.torch.core.nn.modules.loss import SmoothL1Loss
from dragon.vm.torch.core.nn.modules.module import Module
from dragon.vm.torch.core.nn.modules.normalization import GroupNorm
from dragon.vm.torch.core.nn.modules.normalization import LayerNorm
from dragon.vm.torch.core.nn.modules.normalization import LocalResponseNorm
from dragon.vm.torch.core.nn.modules.padding import ConstantPad1d
from dragon.vm.torch.core.nn.modules.padding import ConstantPad2d
from dragon.vm.torch.core.nn.modules.padding import ConstantPad3d
from dragon.vm.torch.core.nn.modules.padding import ReflectionPad1d
from dragon.vm.torch.core.nn.modules.padding import ReflectionPad2d
from dragon.vm.torch.core.nn.modules.padding import ReflectionPad3d
from dragon.vm.torch.core.nn.modules.padding import ReplicationPad1d
from dragon.vm.torch.core.nn.modules.padding import ReplicationPad2d
from dragon.vm.torch.core.nn.modules.padding import ReplicationPad3d
from dragon.vm.torch.core.nn.modules.padding import ZeroPad2d
from dragon.vm.torch.core.nn.modules.pixelshuffle import PixelShuffle
from dragon.vm.torch.core.nn.modules.pixelshuffle import PixelUnshuffle
from dragon.vm.torch.core.nn.modules.pooling import AdaptiveAvgPool1d
from dragon.vm.torch.core.nn.modules.pooling import AdaptiveAvgPool2d
from dragon.vm.torch.core.nn.modules.pooling import AdaptiveAvgPool3d
from dragon.vm.torch.core.nn.modules.pooling import AdaptiveMaxPool1d
from dragon.vm.torch.core.nn.modules.pooling import AdaptiveMaxPool2d
from dragon.vm.torch.core.nn.modules.pooling import AdaptiveMaxPool3d
from dragon.vm.torch.core.nn.modules.pooling import AvgPool1d
from dragon.vm.torch.core.nn.modules.pooling import AvgPool2d
from dragon.vm.torch.core.nn.modules.pooling import AvgPool3d
from dragon.vm.torch.core.nn.modules.pooling import MaxPool1d
from dragon.vm.torch.core.nn.modules.pooling import MaxPool2d
from dragon.vm.torch.core.nn.modules.pooling import MaxPool3d
from dragon.vm.torch.core.nn.modules.rnn import GRU
from dragon.vm.torch.core.nn.modules.rnn import LSTM
from dragon.vm.torch.core.nn.modules.rnn import LSTMCell
from dragon.vm.torch.core.nn.modules.rnn import RNN
from dragon.vm.torch.core.nn.modules.rnn import RNNBase
from dragon.vm.torch.core.nn.modules.rnn import RNNCellBase
from dragon.vm.torch.core.nn.modules.sparse import Embedding
from dragon.vm.torch.core.nn.modules.transformer import TransformerDecoder
from dragon.vm.torch.core.nn.modules.transformer import TransformerDecoderLayer
from dragon.vm.torch.core.nn.modules.transformer import TransformerEncoder
from dragon.vm.torch.core.nn.modules.transformer import TransformerEncoderLayer
from dragon.vm.torch.core.nn.modules.upsampling import Upsample
from dragon.vm.torch.core.nn.modules.upsampling import UpsamplingBilinear2d
from dragon.vm.torch.core.nn.modules.upsampling import UpsamplingNearest2d
from dragon.vm.torch.core.nn.parameter import Parameter

__all__ = [_s for _s in dir() if not _s.startswith('_')]
