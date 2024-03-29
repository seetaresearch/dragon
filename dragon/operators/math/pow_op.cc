#include "dragon/core/workspace.h"
#include "dragon/operators/math/elementwise_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void PowGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1);
  auto &Y = Input(2), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1);

  vec64_t A_broadcast_axes, B_broadcast_axes;
  math::utils::ComputeBroadcastAxes(
      A.dims(), B.dims(), dY.dims(), A_broadcast_axes, B_broadcast_axes);

  // Scratch to save the intermediate data.
  int64_t scratch_size = 0;
  if ((dB->has_name() && !B_broadcast_axes.empty()) ||
      (dA->has_name() && !A_broadcast_axes.empty())) {
    scratch_size = Y.count();
  }
  T* data = ctx()->workspace()->template data<T, Context>(scratch_size);

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
        math::Log(A.count(), A.template data<T, Context>(), data, ctx());
        math::Mul(
            A.ndim(),
            A.dims().data(),
            Y.ndim(),
            Y.dims().data(),
            data,
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
        math::Log(A.count(), A.template data<T, Context>(), data, ctx());
        math::Mul(Y.count(), data, Y.template data<T, Context>(), data, ctx());
      } else {
        math::Log(
            A.count(),
            A.template data<T, Context>(),
            dA->ReshapeLike(A)->template mutable_data<T, Context>(),
            ctx());
        math::Mul(
            A.ndim(),
            A.dims().data(),
            Y.ndim(),
            Y.dims().data(),
            dA->template data<T, Context>(),
            Y.template data<T, Context>(),
            data,
            ctx());
      }
      math::Mul(Y.count(), dY.template data<T, Context>(), data, data, ctx());
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          B_broadcast_axes.size(),
          B_broadcast_axes.data(),
          1.f,
          data,
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
      math::NaNToNum(
          A.count(),
          0.f,
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
      math::Div(
          Y.ndim(),
          Y.dims().data(),
          A.ndim(),
          A.dims().data(),
          Y.template data<T, Context>(),
          A.template data<T, Context>(),
          data,
          ctx());
      math::NaNToNum(Y.count(), 0.f, data, data, ctx());
      if (B_broadcast_axes.empty()) {
        math::Mul(Y.count(), data, B.template data<T, Context>(), data, ctx());
      } else {
        math::Mul(
            Y.ndim(),
            Y.dims().data(),
            B.ndim(),
            B.dims().data(),
            data,
            B.template data<T, Context>(),
            data,
            ctx());
      }
      math::Mul(Y.count(), dY.template data<T, Context>(), data, data, ctx());
      math::ReduceSum(
          dY.ndim(),
          dY.dims().data(),
          A_broadcast_axes.size(),
          A_broadcast_axes.data(),
          1.f,
          data,
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          ctx());
    }
  }
}

template <class Context>
void PowGradientOp<Context>::RunOnDevice() {
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(PowGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(PowGradient);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(PowGradient, PowGradient);
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
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), O(0), GO(0)}),
        vector<string>({GI(0), GI(1)}));
  }
};

} // namespace

REGISTER_GRADIENT(Pow, GradientMaker);

} // namespace dragon
