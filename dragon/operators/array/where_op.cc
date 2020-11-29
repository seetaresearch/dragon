#include "dragon/operators/array/where_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void WhereOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1);
  auto &C = Input(2), *Y = Output(0);

  CHECK(C.template IsType<bool>() || C.template IsType<uint8_t>())
      << "\nExcepted bool or uint8 condition tensor.";

  vec64_t AB_dims, Y_dims;
  if (math::utils::IsBinaryBroadcast(A.dims(), B.dims(), AB_dims) &&
      math::utils::IsBinaryBroadcast(AB_dims, C.dims(), Y_dims)) {
    math::Where(
        A.ndim(),
        A.dims().data(),
        B.ndim(),
        B.dims().data(),
        C.ndim(),
        C.dims().data(),
        A.template data<T, Context>(),
        B.template data<T, Context>(),
        (const bool*)C.template raw_data<Context>(),
        Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Could not broadcast together with shapes: " << A.DimString()
               << " " << B.DimString() << " " << C.DimString();
  }
}

template <class Context>
void WhereOp<Context>::RunOnDevice() {
  DispatchHelper<FullTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void WhereGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &C = Input(2), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1);

  CHECK(C.template IsType<bool>() || C.template IsType<uint8_t>())
      << "\nExcepted bool or uint8 condition tensor.";

  vec32_t A_broadcast_axes, B_broadcast_axes;
  vec32_t Y_dims(dY.dims().begin(), dY.dims().end());
  math::utils::ComputeBinaryBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  // Temporal space to store the intermediate gradient and zeros
  T *scratch = nullptr, *zeros = nullptr;

  // Determine the scratch size
  int64_t scratch_size = 0;
  if (dA->has_name() || dB->has_name()) {
    scratch_size += 1;
    if (!A_broadcast_axes.empty() || !B_broadcast_axes.empty()) {
      scratch_size += dY.count();
    }
  }

  if (scratch_size > 0) {
    scratch = ctx()->workspace()->template data<T, Context>({scratch_size})[0];
    zeros = scratch + (scratch_size - 1);
    math::Set(1, convert::To<T>(0.f), zeros, ctx());
  }

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      math::Where(
          dY.ndim(),
          dY.dims().data(),
          0,
          nullptr,
          C.ndim(),
          C.dims().data(),
          dY.template data<T, Context>(),
          zeros,
          (const bool*)C.template raw_data<Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Where(
          dY.ndim(),
          dY.dims().data(),
          0,
          nullptr,
          C.ndim(),
          C.dims().data(),
          dY.template data<T, Context>(),
          zeros,
          (const bool*)C.template raw_data<Context>(),
          scratch,
          ctx());
      math::ReduceSum(
          Y_dims.size(),
          Y_dims.data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          scratch,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
  }

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      math::Where(
          0,
          nullptr,
          dY.ndim(),
          dY.dims().data(),
          C.ndim(),
          C.dims().data(),
          zeros,
          dY.template data<T, Context>(),
          (const bool*)C.template raw_data<Context>(),
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Where(
          0,
          nullptr,
          dY.ndim(),
          dY.dims().data(),
          C.ndim(),
          C.dims().data(),
          zeros,
          dY.template data<T, Context>(),
          (const bool*)C.template raw_data<Context>(),
          scratch,
          ctx());
      math::ReduceSum(
          Y_dims.size(),
          Y_dims.data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          1.f,
          scratch,
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void WhereGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Where);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Where);
#endif

DEPLOY_CPU_OPERATOR(WhereGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(WhereGradient);
#endif

OPERATOR_SCHEMA(Where)
    /* A, B, C */
    .NumInputs(3)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(WhereGradient)
    /* A, B, C, dY */
    .NumInputs(4)
    /* dA, dB */
    .NumOutputs(2);

namespace {

class GradientMaker : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), I(2), GO(0)}),
        vector<string>({GI(0), GI(1)}));
  }
};

} // namespace

REGISTER_GRADIENT(Where, GradientMaker);

} // namespace dragon
