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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy
try:
    from nvidia.dali import types as dali_types

    # ConstantWrapper
    Constant = dali_types.Constant
    ScalarConstant = dali_types.ScalarConstant

    # DALIDataType
    BOOL = dali_types.BOOL
    FLOAT = dali_types.FLOAT
    FLOAT16 = dali_types.FLOAT16
    FLOAT32 = dali_types.FLOAT
    FLOAT64 = dali_types.FLOAT64
    INT8 = dali_types.INT8
    INT16 = dali_types.INT16
    INT32 = dali_types.INT32
    INT64 = dali_types.INT64
    STRING = dali_types.STRING
    UINT8 = dali_types.UINT8
    UINT16 = dali_types.UINT16
    UINT32 = dali_types.UINT32
    UINT64 = dali_types.UINT64

    # DALIImageType
    BGR = dali_types.BGR
    RGB = dali_types.RGB

    # DALIInterpType
    INTERP_CUBIC = dali_types.INTERP_CUBIC
    INTERP_GAUSSIAN = dali_types.INTERP_GAUSSIAN
    INTERP_LANCZOS3 = dali_types.INTERP_LANCZOS3
    INTERP_LINEAR = dali_types.INTERP_LINEAR
    INTERP_NN = dali_types.INTERP_NN
    INTERP_TRIANGULAR = dali_types.INTERP_TRIANGULAR

    # PipelineAPIType
    PIPELINE_API_BASIC = dali_types.PipelineAPIType.BASIC
    PIPELINE_API_ITERATOR = dali_types.PipelineAPIType.ITERATOR
    PIPELINE_API_SCHEDULED = dali_types.PipelineAPIType.SCHEDULED

    # TensorLayout
    NCHW = dali_types.NCHW
    NHWC = dali_types.NHWC

except ImportError:

    dali_types = None
    NO_DALI = -1

    # ConstantWrapper
    Constant = NO_DALI
    ScalarConstant = NO_DALI

    # DALIDataType
    BOOL = NO_DALI
    FLOAT = NO_DALI
    FLOAT16 = NO_DALI
    FLOAT32 = NO_DALI
    FLOAT64 = NO_DALI
    INT8 = NO_DALI
    INT16 = NO_DALI
    INT32 = NO_DALI
    INT64 = NO_DALI
    STRING = NO_DALI
    UINT8 = NO_DALI
    UINT16 = NO_DALI
    UINT32 = NO_DALI
    UINT64 = NO_DALI

    # DALIImageType
    BGR = NO_DALI
    RGB = NO_DALI

    # DALIInterpType
    INTERP_CUBIC = NO_DALI
    INTERP_GAUSSIAN = NO_DALI
    INTERP_LANCZOS3 = NO_DALI
    INTERP_LINEAR = NO_DALI
    INTERP_NN = NO_DALI
    INTERP_TRIANGULAR = NO_DALI

    # PipelineAPIType
    PIPELINE_API_BASIC = NO_DALI
    PIPELINE_API_ITERATOR = NO_DALI
    PIPELINE_API_SCHEDULED = NO_DALI

    # TensorLayout
    NCHW = NO_DALI
    NHWC = NO_DALI


def np_dtype(dali_dtype):
    """Convert the dali dtype into the numpy format."""
    return numpy.dtype(dali_dtype)
