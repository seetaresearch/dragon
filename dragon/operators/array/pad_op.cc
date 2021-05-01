#include "dragon/operators/array/pad_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void PadOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_pads, num_dims = X.ndim();
  pads(0, &num_pads);

  CHECK_EQ(num_pads, num_dims * 2)
      << "\nGot " << num_pads << " pads (" << num_pads / 2 << " axes) "
      << "for input dimensions " << X.DimString() << ".";

  vec64_t X_pads(num_dims * 2), Y_dims(X.dims());

  for (int i = 0; i < num_dims; ++i) {
    X_pads[i] = pads(i);
    X_pads[i + num_dims] = pads(i + num_dims);
    Y_dims[i] += (X_pads[i] + X_pads[i + num_dims]);
  }

  // Store for the gradient calculation
  Buffer("X_pads")->template CopyFrom<int64_t>(X_pads);

  if (X.dims() == Y_dims) {
    Y->Reshape(Y_dims)->CopyFrom(X, ctx());
    return; // Just copy the contents
  }

  if (mode_ == "CONSTANT") {
    kernels::ConstPad(
        X.ndim(),
        X.dims().data(),
        X.strides().data(),
        Y_dims.data(),
        X_pads.data(),
        value_,
        X.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else if (mode_ == "REFLECT") {
    for (int i = 0; i < num_dims; ++i) {
      CHECK_LE(X_pads[i], X.dim(i) + 1)
          << "\nThe dimension of axis " << i << " is " << X.dim(i) << ","
          << "\nwhile the excepted begin of padding"
          << "for reflect mode should be in (0, " << X.dim(i) + 1 << "].";
      CHECK_LE(X_pads[i + num_dims], X.dim(i) - 1)
          << "\nThe dimension of axis " << i << " is " << X.dim(i) << ","
          << "\nwhile the excepted end of padding "
          << "for reflect mode should be in (0, " << X.dim(i) - 1 << "].";
    }
    kernels::ReflectPad(
        X.ndim(),
        X.dims().data(),
        X.strides().data(),
        Y_dims.data(),
        X_pads.data(),
        X.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else if (mode_ == "EDGE") {
    kernels::EdgePad(
        X.ndim(),
        X.dims().data(),
        X.strides().data(),
        Y_dims.data(),
        X_pads.data(),
        X.template data<T, Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unknown PadMode: " << mode_ << ".";
  }
}

template <class Context>
template <typename T>
void PadGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);

  vec64_t X_dims(dY.dims()), X_pads;
  Buffer("X_pads")->template CopyTo<int64_t>(X_pads);

  // Restore the input dimensions
  int num_dims = dY.ndim();
  for (int i = 0; i < num_dims; ++i)
    X_dims[i] -= (X_pads[i] + X_pads[i + num_dims]);

  if (dY.dims() == X_dims) {
    dX->Reshape(X_dims)->CopyFrom(dY, ctx());
    return; // Just copy the contents
  }

  if (mode_ == "CONSTANT") {
    kernels::Slice(
        num_dims,
        dY.strides().data(),
        X_dims.data(),
        X_pads.data(),
        dY.template data<T, Context>(),
        dX->Reshape(X_dims)->template mutable_data<T, Context>(),
        ctx());
  } else if (mode_ == "REFLECT") {
    LOG(FATAL) << "No implementation for <ReflectPadGrad>";
  } else if (mode_ == "EDGE") {
    LOG(FATAL) << "No implementation for <EdgePadGrad>";
  } else {
    LOG(FATAL) << "Unknown PadMode: " << mode_ << ".";
  }
}

DEPLOY_CPU_OPERATOR(Pad);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Pad);
#endif

DEPLOY_CPU_OPERATOR(PadGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(PadGradient);
#endif

OPERATOR_SCHEMA(Pad)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(PadGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1);

REGISTER_GRADIENT(Pad, SimpleGradientMaker);

} // namespace dragon
