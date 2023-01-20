#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/array/pad_op.h"
#include "dragon/utils/conversions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLPadOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_pads, num_dims = X.ndim();
  pads(0, &num_pads);

  CHECK_EQ(num_pads, num_dims * 2)
      << "\nGot " << num_pads << " pads (" << num_pads / 2 << " axes) "
      << "for input dimensions " << X.DimString() << ".";

  vec64_t X_pads(num_dims * 2), Y_dims(X.dims());
  vec32_t paddings(num_dims * 2);
  for (int i = 0; i < num_dims; ++i) {
    X_pads[i] = pads(i);
    X_pads[i + num_dims] = pads(i + num_dims);
    Y_dims[i] += (X_pads[i] + X_pads[i + num_dims]);
    paddings[i * 2] = X_pads[i];
    paddings[i * 2 + 1] = X_pads[i + num_dims];
  }

  // Save for the gradient computation.
  Output("X_pads")->template CopyFrom<int64_t>(X_pads);

  if (X.dims() == Y_dims) {
    Y->Reshape(Y_dims)->CopyFrom(X, ctx());
    return;
  }

  CNNLSetTensorDesc<T>(input_desc_, X.dims());
  CNNLSetTensorDesc<T>(output_desc_, Y_dims);

  if (mode_ == "CONSTANT") {
    const auto value_fpcast = convert::To<T>(value_);
    CNNL_CHECK(cnnlPad(
        ctx()->cnnl_handle(),
        input_desc_,
        X.template data<T, Context>(),
        paddings.data(),
        &value_fpcast,
        output_desc_,
        Y->Reshape(Y_dims)->template mutable_data<T, Context>()));
  } else {
    LOG(FATAL) << "Unsupported Pad mode: " << mode_;
  }
}

template <class Context>
template <typename T>
void CNNLPadGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(1), *dX = Output(0);

  int num_dims = dY.ndim();
  vec64_t X_dims(dY.dims()), X_pads, X_ends(num_dims);
  Input("X_pads").template CopyTo<int64_t>(X_pads);

  // Restore the input dimensions
  for (int i = 0; i < num_dims; ++i) {
    X_dims[i] -= (X_pads[i] + X_pads[i + num_dims]);
    X_ends[i] = X_pads[i] + X_dims[i];
  }

  if (dY.dims() == X_dims) {
    dX->Reshape(X_dims)->CopyFrom(dY, ctx());
    return;
  }

  CNNLSetTensorDesc<T>(input_desc_, dY.dims());
  CNNLSetTensorDesc<T>(output_desc_, X_dims);

  if (mode_ == "CONSTANT") {
    CNNL_CHECK(cnnlStridedSlice(
        ctx()->cnnl_handle(),
        input_desc_,
        dY.template data<T, Context>(),
        vec32_t({X_pads.begin(), X_pads.end()}).data(),
        vec32_t({X_ends.begin(), X_ends.end()}).data(),
        vec32_t(num_dims, 1).data(),
        output_desc_,
        dX->Reshape(X_dims)->template mutable_data<T, Context>()));
  } else {
    LOG(FATAL) << "Unsupported Pad mode: " << mode_;
  }
}

DEPLOY_CNNL_OPERATOR(Pad);
DEPLOY_CNNL_OPERATOR(PadGradient);

DEFINE_OP_REPEATED_ARG(int64_t, CNNLPadOp, pads);

} // namespace dragon

#endif // USE_MLU
