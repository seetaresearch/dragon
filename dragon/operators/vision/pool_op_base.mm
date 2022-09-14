#include "dragon/operators/vision/pool_op_base.h"

namespace dragon {

template <class Context>
MPSPoolOpBase<Context>::MPSPoolOpBase(const OperatorDef& def, Workspace* ws)
    : PoolOpBase<Context>(def, ws) {
  GetBaseArguments();
  pool2d_desc_ = [[MPSGraphPooling2DOpDescriptor new] autorelease];
}

template <class Context>
void MPSPoolOpBase<Context>::SetPoolDesc() {
  if (num_axes_ == 1 || num_axes_ == 2) {
    pool2d_desc_.kernelHeight = kshape_[0];
    pool2d_desc_.kernelWidth = num_axes_ == 1 ? 1 : kshape_[1];
    pool2d_desc_.strideInY = strides_[0];
    pool2d_desc_.strideInX = num_axes_ == 1 ? 1 : strides_[1];
    pool2d_desc_.paddingTop = pads_begin_[0];
    pool2d_desc_.paddingBottom = pads_end_[0];
    pool2d_desc_.paddingLeft = num_axes_ == 1 ? 0 : pads_begin_[1];
    pool2d_desc_.paddingRight = num_axes_ == 1 ? 0 : pads_end_[1];
    pool2d_desc_.dilationRateInY = 1;
    pool2d_desc_.dilationRateInX = 1;
    pool2d_desc_.ceilMode = ceil_mode_ > 0;
    pool2d_desc_.paddingStyle = MPSGraphPaddingStyleExplicit;
    pool2d_desc_.includeZeroPadToAverage = true; // macOS 12.0 or higher.
    if (data_format() == "NCHW") {
      pool2d_desc_.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
    } else if (data_format() == "NHWC") {
      pool2d_desc_.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
    } else {
      LOG(FATAL) << "Unsupported DataFormat: " << data_format();
    }
  }
}

template class MPSPoolOpBase<MPSContext>;

} // namespace dragon
