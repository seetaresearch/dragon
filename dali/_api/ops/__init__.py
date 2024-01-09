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
# ------------------------------------------------------------------------

from dragon.vm.dali.core.ops.bbox_ops import BbFlip
from dragon.vm.dali.core.ops.bbox_ops import BBoxPaste
from dragon.vm.dali.core.ops.builtin_ops import ExternalSource
from dragon.vm.dali.core.ops.decoder_ops import ImageDecoder
from dragon.vm.dali.core.ops.decoder_ops import ImageDecoderRandomCrop
from dragon.vm.dali.core.ops.generic_ops import Cast
from dragon.vm.dali.core.ops.generic_ops import Erase
from dragon.vm.dali.core.ops.generic_ops import Flip
from dragon.vm.dali.core.ops.generic_ops import Pad
from dragon.vm.dali.core.ops.generic_ops import Reshape
from dragon.vm.dali.core.ops.generic_ops import Slice
from dragon.vm.dali.core.ops.image_ops import Brightness
from dragon.vm.dali.core.ops.image_ops import BrightnessContrast
from dragon.vm.dali.core.ops.image_ops import Contrast
from dragon.vm.dali.core.ops.image_ops import ColorSpaceConversion
from dragon.vm.dali.core.ops.image_ops import ColorTwist
from dragon.vm.dali.core.ops.image_ops import CropMirrorNormalize
from dragon.vm.dali.core.ops.image_ops import GaussianBlur
from dragon.vm.dali.core.ops.image_ops import Hsv
from dragon.vm.dali.core.ops.image_ops import Paste
from dragon.vm.dali.core.ops.image_ops import RandomBBoxCrop
from dragon.vm.dali.core.ops.image_ops import RandomResizedCrop
from dragon.vm.dali.core.ops.image_ops import Resize
from dragon.vm.dali.core.ops.image_ops import Rotate
from dragon.vm.dali.core.ops.image_ops import WarpAffine
from dragon.vm.dali.core.ops.math_ops import Normalize
from dragon.vm.dali.core.ops.random_ops import CoinFlip
from dragon.vm.dali.core.ops.random_ops import Uniform
from dragon.vm.dali.core.ops.reader_ops import TFRecordReader
