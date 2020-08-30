#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_ops.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void PowGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &Y = Input(2), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1);

  vec32_t A_broadcast_axes, B_broadcast_axes;
  vec32_t Y_dims(dY.dims().begin(), dY.dims().end());
  utils::math::ComputeBinaryBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  // Temporal space to store the intermediate gradient
  T* scratch = nullptr;

  if (dB->has_name()) {
    if (B_broadcast_axes.empty()) {
      if (A_broadcast_axes.empty()) {
        math::Log(
            A.count(),
            A.template data<T, Context>(),
            dB->ReshapeLike(B)->template mutable_data<T, Context>(),
            ctx());
        math::Mul(
            Y.count(),
            dB->template data<T, Context>(),
            Y.template data<T, Context>(),
            dB->template mutable_data<T, Context>(),
            ctx());
      } else {
        scratch = ws()->template data<T, Context>({dY.count()})[0];
        math::Log(A.count(), A.template data<T, Context>(), scratch, ctx());
        math::Mul(
            A.ndim(),
            A.dims().data(),
            Y.ndim(),
            Y.dims().data(),
            scratch,
            Y.template data<T, Context>(),
            dB->ReshapeLike(B)->template mutable_data<T, Context>(),
            ctx());
      }
      math::Mul(
          Y.count(),
          dY.template data<T, Context>(),
          dB->template data<T, Context>(),
          dB->template mutable_data<T, Context>(),
          ctx());
    } else {
      if (A_broadcast_axes.empty()) {
        scratch = ws()->template data<T, Context>({dY.count()})[0];
        math::Log(A.count(), A.template data<T, Context>(), scratch, ctx());
        math::Mul(
            Y.count(), scratch, Y.template data<T, Context>(), scratch, ctx());
      } else {
        auto scratches =
            ws()->template data<T, Context>({dY.count(), A.count()});
        scratch = scratches[0];
        math::Log(
            A.count(), A.template data<T, Context>(), scratches[1], ctx());
        math::Mul(
            A.ndim(),
            A.dims().data(),
            Y.ndim(),
            Y.dims().data(),
            scratches[1],
            Y.template data<T, Context>(),
            scratch,
            ctx());
      }
      math::Mul(
          Y.count(), dY.template data<T, Context>(), scratch, scratch, ctx());
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

  if (dA->has_name()) {
    if (A_broadcast_axes.empty()) {
      math::Div(
          A.count(),
          Y.template data<T, Context>(),
          A.template data<T, Context>(),
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
      math::ReplaceNaN(
          A.count(),
          cast::to<T>(0.f),
          dA->template data<T, Context>(),
          dA->template mutable_data<T, Context>(),
          ctx());
      if (B_broadcast_axes.empty()) {
        math::Mul(
            B.count(),
            dA->template data<T, Context>(),
            B.template data<T, Context>(),
            dA->template mutable_data<T, Context>(),
            ctx());
      } else {
        math::Mul(
            A.ndim(),
            A.dims().data(),
            B.ndim(),
            B.dims().data(),
            dA->template data<T, Context>(),
            B.template data<T, Context>(),
            dA->template mutable_data<T, Context>(),
            ctx());
      }
      math::Mul(
          A.count(),
          dY.template data<T, Context>(),
          dA->template data<T, Context>(),
          dA->template mutable_data<T, Context>(),
          ctx());
    } else {
      if (scratch == nullptr) {
        scratch = ws()->template data<T, Context>({dY.count()})[0];
      }
      math::Div(
          Y.ndim(),
          Y.dims().data(),
          A.ndim(),
          A.dims().data(),
          Y.template data<T, Context>(),
          A.template data<T, Context>(),
          scratch,
          ctx());
      math::ReplaceNaN(Y.count(), cast::to<T>(0.f), scratch, scratch, ctx());
      if (B_broadcast_axes.empty()) {
        math::Mul(
            Y.count(), scratch, B.template data<T, Context>(), scratch, ctx());
      } else {
        math::Mul(
            Y.ndim(),
            Y.dims().data(),
            B.ndim(),
            B.dims().data(),
            scratch,
            B.template data<T, Context>(),
            scratch,
            ctx());
      }
      math::Mul(
          Y.count(), dY.template data<T, Context>(), scratch, scratch, ctx());
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
}

template <class Context>
void PowGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(PowGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(PowGradient);
#endif

OPERATOR_SCHEMA(PowGradient)
    /* A, B, Y, dY */
    .NumInputs(4)
    /* dA, dB */
    .NumOutputs(2);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({I(0), I(1), O(0), GO(0)}),
        vector<string>({GI(0), GI(1)}));
  }
};

} // namespace

REGISTER_GRADIENT(Pow, GradientMaker);

} // namespace dragon
