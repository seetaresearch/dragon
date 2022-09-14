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

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.keras.core.engine.base_layer import Layer
from dragon.vm.keras.core.engine.input_spec import InputSpec
from dragon.vm.keras.core.layers.activation import ELU
from dragon.vm.keras.core.layers.activation import LeakyReLU
from dragon.vm.keras.core.layers.activation import ReLU
from dragon.vm.keras.core.layers.activation import SELU
from dragon.vm.keras.core.layers.activation import Softmax
from dragon.vm.keras.core.layers.convolutional import Conv1D
from dragon.vm.keras.core.layers.convolutional import Conv1D as Convolution1D
from dragon.vm.keras.core.layers.convolutional import Conv2D
from dragon.vm.keras.core.layers.convolutional import Conv2D as Convolution2D
from dragon.vm.keras.core.layers.convolutional import Conv3D
from dragon.vm.keras.core.layers.convolutional import Conv3D as Convolution3D
from dragon.vm.keras.core.layers.convolutional import Conv1DTranspose
from dragon.vm.keras.core.layers.convolutional import Conv1DTranspose as Convolution1DTranspose
from dragon.vm.keras.core.layers.convolutional import Conv2DTranspose
from dragon.vm.keras.core.layers.convolutional import Conv2DTranspose as Convolution2DTranspose
from dragon.vm.keras.core.layers.convolutional import Conv3DTranspose
from dragon.vm.keras.core.layers.convolutional import Conv3DTranspose as Convolution3DTranspose
from dragon.vm.keras.core.layers.convolutional import DepthwiseConv2D
from dragon.vm.keras.core.layers.core import Activation
from dragon.vm.keras.core.layers.core import Dense
from dragon.vm.keras.core.layers.core import Dropout
from dragon.vm.keras.core.layers.merging import Add
from dragon.vm.keras.core.layers.merging import Concatenate
from dragon.vm.keras.core.layers.merging import Maximum
from dragon.vm.keras.core.layers.merging import Minimum
from dragon.vm.keras.core.layers.merging import Multiply
from dragon.vm.keras.core.layers.merging import Subtract
from dragon.vm.keras.core.layers.normalization import BatchNormalization
from dragon.vm.keras.core.layers.normalization import LayerNormalization
from dragon.vm.keras.core.layers.pooling import AveragePooling1D
from dragon.vm.keras.core.layers.pooling import AveragePooling1D as AvgPool1D
from dragon.vm.keras.core.layers.pooling import AveragePooling2D
from dragon.vm.keras.core.layers.pooling import AveragePooling2D as AvgPool2D
from dragon.vm.keras.core.layers.pooling import AveragePooling3D
from dragon.vm.keras.core.layers.pooling import AveragePooling3D as AvgPool3D
from dragon.vm.keras.core.layers.pooling import GlobalAveragePooling1D
from dragon.vm.keras.core.layers.pooling import GlobalAveragePooling1D as GlobalAvgPool1D
from dragon.vm.keras.core.layers.pooling import GlobalAveragePooling2D
from dragon.vm.keras.core.layers.pooling import GlobalAveragePooling2D as GlobalAvgPool2D
from dragon.vm.keras.core.layers.pooling import GlobalAveragePooling3D
from dragon.vm.keras.core.layers.pooling import GlobalAveragePooling3D as GlobalAvgPool3D
from dragon.vm.keras.core.layers.pooling import GlobalMaxPooling1D
from dragon.vm.keras.core.layers.pooling import GlobalMaxPooling1D as GlobalMaxPool1D
from dragon.vm.keras.core.layers.pooling import GlobalMaxPooling2D
from dragon.vm.keras.core.layers.pooling import GlobalMaxPooling2D as GlobalMaxPool2D
from dragon.vm.keras.core.layers.pooling import GlobalMaxPooling3D
from dragon.vm.keras.core.layers.pooling import GlobalMaxPooling3D as GlobalMaxPool3D
from dragon.vm.keras.core.layers.pooling import MaxPooling1D
from dragon.vm.keras.core.layers.pooling import MaxPooling1D as MaxPool1D
from dragon.vm.keras.core.layers.pooling import MaxPooling2D
from dragon.vm.keras.core.layers.pooling import MaxPooling2D as MaxPool2D
from dragon.vm.keras.core.layers.pooling import MaxPooling3D
from dragon.vm.keras.core.layers.pooling import MaxPooling3D as MaxPool3D
from dragon.vm.keras.core.layers.reshaping import Flatten
from dragon.vm.keras.core.layers.reshaping import Permute
from dragon.vm.keras.core.layers.reshaping import Reshape
from dragon.vm.keras.core.layers.reshaping import UpSampling1D
from dragon.vm.keras.core.layers.reshaping import UpSampling2D
from dragon.vm.keras.core.layers.reshaping import UpSampling3D
from dragon.vm.keras.core.layers.reshaping import ZeroPadding1D
from dragon.vm.keras.core.layers.reshaping import ZeroPadding2D
from dragon.vm.keras.core.layers.reshaping import ZeroPadding3D

__all__ = [_s for _s in dir() if not _s.startswith('_')]
