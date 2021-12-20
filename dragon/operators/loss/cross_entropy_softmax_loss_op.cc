#include "dragon/core/workspace.h"
#include "dragon/operators/loss/cross_entropy_loss_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void SoftmaxCrossEntropyLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  const auto NxCxS = X.count();

  T *loss = nullptr, *mask = nullptr;
  auto* X_norm = Output("X_norm")->ReshapeLike(X);
  auto* input = X_norm->template mutable_data<T, Context>();

  kernels::Softmax(N, S, C, X.template data<T, Context>(), input, ctx());
  if (Y.meta() == TypeMeta::Make<T>()) {
    CHECK_EQ(Y.count(), NxCxS) << "\nNumel of X and Y must be matched.";
    auto* target = Y.template data<T, Context>();
    loss = ctx()->workspace()->template data<T, Context>(NxCxS);
    kernels::CrossEntropy(NxCxS, input, target, loss, ctx());
  } else {
    CHECK_EQ(Y.count(), NxS) << "\nNumel of X and Y must be matched.";
    auto* scratch = ctx()->workspace()->template data<T, Context>(NxS * 2 + 1);
    loss = scratch, mask = scratch + NxS;
    if (Y.meta() == TypeMeta::Make<int>()) {
      auto* target = Y.template data<int, Context>();
      kernels::CrossEntropy(
          N, S, C, ignore_index_, input, target, loss, mask, ctx());
    } else if (Y.meta() == TypeMeta::Make<int64_t>()) {
      auto* target = Y.template data<int64_t, Context>();
      kernels::CrossEntropy(
          N, S, C, ignore_index_, input, target, loss, mask, ctx());
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(Y.meta()),
          {"int32", "int64", dtypes::to_string<T>()});
    }
  }

  if (reduction_ == "NONE") {
    auto out_dims = X.dims();
    out_dims.erase(out_dims.begin() + axis);
    if (mask == nullptr) {
      math::ReduceSum(
          3,
          vec64_t({N, C, S}).data(),
          1,
          vec64_t({1}).data(),
          1.f,
          loss,
          L->Reshape(out_dims)->template mutable_data<T, Context>(),
          ctx());
    } else {
      math::Copy(
          NxS,
          loss,
          L->Reshape(out_dims)->template mutable_data<T, Context>(),
          ctx());
    }
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = (mask == nullptr ? NxS : -1);
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = NxS;
    }
    kernels::ReduceLoss(
        mask == nullptr ? NxCxS : NxS,
        mask == nullptr ? 0 : NxS,
        normalizer,
        loss,
        mask,
        L->Reshape({})->template mutable_data<T, Context>(),
        ctx());
  }
}

template <class Context>
template <typename T>
void SoftmaxCrossEntropyLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dL = Input(2);
  auto* dX = Output(0)->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  const auto NxCxS = X.count();

  T* mask = nullptr;
  auto* input = Input("X_norm").template data<T, Context>();
  auto* dl = dL.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();

  math::Copy(NxCxS, input, dx, ctx());
  if (Y.meta() == TypeMeta::Make<T>()) {
    auto* target = Y.template data<T, Context>();
    math::Axpy(NxCxS, -1.f, target, dx, ctx());
  } else {
    mask = ctx()->workspace()->template data<T, Context>(NxS + 1);
    if (Y.meta() == TypeMeta::Make<int>()) {
      auto* target = Y.template data<int, Context>();
      kernels::SoftmaxCrossEntropyGrad(
          N, S, C, ignore_index_, input, target, dx, mask, ctx());
    } else if (Y.meta() == TypeMeta::Make<int64_t>()) {
      auto* target = Y.template data<int64_t, Context>();
      kernels::SoftmaxCrossEntropyGrad(
          N, S, C, ignore_index_, input, target, dx, mask, ctx());
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(Y.meta()),
          {"int32", "int64", dtypes::to_string<T>()});
    }
  }

  if (reduction_ == "NONE") {
    kernels::BroadcastLossGrad(N, S, C, dl, dx, ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "VALID") {
      normalizer = (mask == nullptr ? NxS : -1);
    } else if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = NxS;
    }
    kernels::ReduceLossGrad(
        NxCxS, mask == nullptr ? 0 : NxS, normalizer, dl, mask, dx, ctx());
  }
}

DEPLOY_CPU_OPERATOR(SoftmaxCrossEntropyLoss);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SoftmaxCrossEntropyLoss);
#endif

DEPLOY_CPU_OPERATOR(SoftmaxCrossEntropyLossGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(SoftmaxCrossEntropyLossGradient);
#endif

OPERATOR_SCHEMA(SoftmaxCrossEntropyLoss)
    /* X, Y */
    .NumInputs(2)
    /* L */
    .NumOutputs(1);

OPERATOR_SCHEMA(SoftmaxCrossEntropyLossGradient)
    /* X, Y, dL */
    .NumInputs(3)
    /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(SoftmaxCrossEntropyLoss, GradientMaker);

} // namespace dragon
