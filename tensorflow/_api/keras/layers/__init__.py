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

from __future__ import absolute_import as _absolute_import
from __future__ import division as _division
from __future__ import print_function as _print_function

from dragon.vm.tensorflow.core.keras.engine.base_layer import Layer
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import ELU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import LeakyReLU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import ReLU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import SELU
from dragon.vm.tensorflow.core.keras.layers.advanced_activations import Softmax
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2D as Convolution2D
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import Conv2DTranspose as Convolution2DTranspose
from dragon.vm.tensorflow.core.keras.layers.convolutional import DepthwiseConv2D
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
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling2D
from dragon.vm.tensorflow.core.keras.layers.pooling import AveragePooling2D as AvgPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalAveragePooling2D as GlobalAvgPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import GlobalMaxPool2D as GlobalMaxPooling2D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPool2D
from dragon.vm.tensorflow.core.keras.layers.pooling import MaxPool2D as MaxPooling2D

__all__ = [_s for _s in dir() if not _s.startswith('_')]
