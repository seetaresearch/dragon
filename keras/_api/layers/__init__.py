# ------------------------------------------------------------------------
# Copyright (c) 2017-present, SeetaTech. All Rights Reserved.
#
# Licensed under the BSD 2-Clause License,
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://opensource.org/licenses/BSD-2-Clause
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# -------------------------------------------------------------------------

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
