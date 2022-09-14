#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void WhereOp<Context>::DoRunWithType() {
  auto &C = Input(0), &A = Input(1), &B = Input(2);
  Output("A_spec")->ReshapeLike(A);
  Output("B_spec")->ReshapeLike(B);

  CHECK(C.template IsType<bool>() || C.template IsType<uint8_t>())
      << "\nExcepted bool or uint8 condition tensor.";

  vec64_t AB_dims, Y_dims;
  if (math::utils::IsBinaryBroadcast(A.dims(), B.dims(), AB_dims) &&
      math::utils::IsBinaryBroadcast(AB_dims, C.dims(), Y_dims)) {
    auto* Y = Output(0, CheckOutputAliases(A, B, Output(0), Y_dims));
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
template <typename T>
void WhereGradientOp<Context>::DoRunWithType() {
  auto &C = Input(0), &dY = Input(1);
  auto &A_spec = Input("A_spec"), &B_spec = Input("B_spec");
  auto *dA = Output(0), *dB = Output(1);

  CHECK(C.template IsType<bool>() || C.template IsType<uint8_t>())
      << "\nExcepted bool or uint8 condition tensor.";

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A_spec.dims(),
      B_spec.dims(),
      dY.dims(),
      A_broadcast_axes,
      B_broadcast_axes);

  // Scratch to save the intermediates.
  int64_t scratch_size = 0, scratch_offset = 0;
  if (dA->has_name() || dB->has_name()) scratch_size += 1;
  if ((dA->has_name() && !A_broadcast_axes.empty()) ||
      (dB->has_name() && !B_broadcast_axes.empty())) {
    scratch_size += (scratch_offset = dY.count());
  }
  auto* scratch = ctx()->workspace()->template data<T, Context>(scratch_size);
  auto* zeros = scratch + scratch_offset;
  if (scratch_size > 0) {
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
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
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
          dY.ndim(),
          dY.dims().data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          scratch,
          dA->ReshapeLike(A_spec)->template mutable_data<T, Context>(),
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
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
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
          dY.ndim(),
          dY.dims().data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          1.f,
          scratch,
          dB->ReshapeLike(B_spec)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void WhereOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Generic>::Call(this, Input(1));
}

template <class Context>
void WhereGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(1));
}

DEPLOY_CPU_OPERATOR(Where);
DEPLOY_CPU_OPERATOR(WhereGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Where);
DEPLOY_CUDA_OPERATOR(WhereGradient);
#endif

OPERATOR_SCHEMA(Where)
    /* C, A, B */
    .NumInputs(3)
    /* Y */
    .NumOutputs(1)
    /* A => Y, B => Y */
    .AllowInplace({{1, 0}, {2, 0}});

OPERATOR_SCHEMA(WhereGradient)
    /* C, dY */
    .NumInputs(2)
    /* dA, dB */
    .NumOutputs(2)
    /* dY => dA, dY => dB */
    .AllowInplace({{1, 0}, {1, 1}});

namespace {

class GradientMaker : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), GO(0)}),
        vector<string>({GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(Where, GradientMaker);

} // namespace dragon
