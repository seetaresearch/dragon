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

__all__ = [_s for _s in dir() if not _s.startswith('_')]
