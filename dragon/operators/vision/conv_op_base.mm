#include "dragon/operators/vision/conv_op_base.h"

namespace dragon {

template <class Context>
MPSConvOpBase<Context>::MPSConvOpBase(const OperatorDef& def, Workspace* ws)
    : ConvOpBase<Context>(def, ws) {
  GetBaseArguments();
  conv2d_desc_ = [[MPSGraphConvolution2DOpDescriptor new] autorelease];
}

template <class Context>
void MPSConvOpBase<Context>::SetConvDesc() {
  if (num_axes_ == 1 || num_axes_ == 2) {
    conv2d_desc_.strideInY = strides_[0];
    conv2d_desc_.strideInX = num_axes_ == 1 ? 1 : strides_[1];
    conv2d_desc_.paddingTop = pads_begin_[0];
    conv2d_desc_.paddingBottom = pads_end_[0];
    conv2d_desc_.paddingLeft = num_axes_ == 1 ? 0 : pads_begin_[1];
    conv2d_desc_.paddingRight = num_axes_ == 1 ? 0 : pads_end_[1];
    conv2d_desc_.dilationRateInY = dilations_[0];
    conv2d_desc_.dilationRateInX = num_axes_ == 1 ? 1 : dilations_[1];
    conv2d_desc_.groups = group_;
    conv2d_desc_.paddingStyle = MPSGraphPaddingStyleExplicit;
    if (data_format() == "NCHW") {
      conv2d_desc_.dataLayout = MPSGraphTensorNamedDataLayoutNCHW;
      conv2d_desc_.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    } else if (data_format() == "NHWC") {
      conv2d_desc_.dataLayout = MPSGraphTensorNamedDataLayoutNHWC;
      // Note: OHWI is not supported, disable unittest currently.
      // conv2d_desc_.weightsLayout = MPSGraphTensorNamedDataLayoutOHWI;
      conv2d_desc_.weightsLayout = MPSGraphTensorNamedDataLayoutOIHW;
    } else {
      LOG(FATAL) << "Unsupported DataFormat: " << data_format();
    }
  }
}

template class MPSConvOpBase<MPSContext>;

} // namespace dragon
