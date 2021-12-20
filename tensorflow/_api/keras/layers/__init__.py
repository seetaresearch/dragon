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

from dragon.vm.tensorflow.core.keras.engine.base_layer import Layer
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import ELU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import LeakyReLU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import ReLU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import SELU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import Softmax
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv1D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv1D as Convolution1D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2D as Convolution2D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv3D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv3D as Convolution3D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv1DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv1DTranspose as Convolution1DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2DTranspose as Convolution2DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv3DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv3DTranspose as Convolution3DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import DepthwiseConv2D
from dragon.vm.tensorflow.core.keras.layers.convolutional import UpSampling1D
from dragon.vm.tensorflow.core.keras.layers.convolutional import UpSampling2D
from dragon.vm.tensorflow.core.keras.layers.convolutional import UpSampling3D
from dragon.vm.tensorflow.core.keras.layers.convolutional import ZeroPadding1D
from dragon.vm.tensorflow.core.keras.layers.convolutional import ZeroPadding2D
from dragon.vm.tensorflow.core.keras.layers.convolutional import ZeroPadding3D
from dragon.vm.tensorflow.core.keras.layers.core import Activation
from dragon.vm.tensorflow.core.keras.layers.core import Dense
from dragon.vm.tensorflow.core.keras.layers.core import Dropout
from dragon.vm.tensorflow.core.keras.layers.core import Flatten
from dragon.vm.tensorflow.core.keras.layers.core import Permute
from dragon.vm.tensorflow.core.keras.layers.core import Reshape
from dragon.vm.tensorflow.core.keras.layers.merge import Add
from dragon.vm.tensorflow.core.keras.layers.merge import Concatenate
from dragon.vm.tensorflow.core.keras.layers.merge import Maximum
from dragon.vm.tensorflow.core.keras.layers.merge import Minimum
from dragon.vm.tensorflow.core.keras.layers.merge import Multiply
from dragon.vm.tensorflow.core.keras.layers.merge import Subtract
from dragon.vm.tensorflow.core.keras.layers.normalization import BatchNormalization
from dragon.vm.tensorflow.core.keras.layers.normalization import LayerNormalization
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling1D
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling1D as AvgPool1D
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling2D
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling2D as AvgPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling3D
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling3D as AvgPool3D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling1D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling1D as GlobalAvgPool1D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling2D as GlobalAvgPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling3D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling3D as GlobalAvgPool3D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPooling1D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPooling1D as GlobalMaxPool1D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPooling2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPooling2D as GlobalMaxPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPooling3D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPooling3D as GlobalMaxPool3D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPooling1D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPooling1D as MaxPool1D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPooling2D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPooling2D as MaxPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPooling3D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPooling3D as MaxPool3D

__all__ = [_s for _s in dir() if not _s.startswith('_')]
