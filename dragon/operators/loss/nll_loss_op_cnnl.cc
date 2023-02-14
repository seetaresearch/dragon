#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/operators/loss/nll_loss_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLNLLLossOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *L = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  const auto X_dims = vec64_t({N, S == 1 ? 1 : C, S == 1 ? C : S});

  auto* input = X.template mutable_data<T, Context>();
  auto* target = Y.template data<int, Context>();
  auto* mutable_target = const_cast<int*>(target);
  CHECK_EQ(Y.count(), NxS) << "\nNumel of X and Y must be matched.";
  auto* loss = reinterpret_cast<T*>(ctx()->workspace()->template data<Context>(
      NxS * (sizeof(T) + sizeof(int)), "BufferKernel"));
  if (ignore_index_ != INT_MAX) {
    CHECK_EQ(S, 1) << "\nSpatial loss does not support <ignore_index>.";
    mutable_target = reinterpret_cast<int*>(loss + NxS);
    kernels::Clip(NxS, 0.f, float(C - 1), target, mutable_target, ctx());
  }

  CNNLSetTensorDesc<T>(input_desc_, {N, C, S});
  CNNLSetTensorDesc<int>(target_desc_, {N, 1, S});
  CNNLSetTensorDesc<T>(output_desc_, {N, 1, S});
  CNNL_CHECK(cnnlGather(
      ctx()->cnnl_handle(),
      1,
      input_desc_,
      input,
      target_desc_,
      target,
      output_desc_,
      loss));
  math::Neg(NxS, loss, loss, ctx());
  kernels::MaskLoss(NxS, ignore_index_, target, loss, ctx());

  if (reduction_ == "NONE") {
    auto out_dims = X.dims();
    out_dims.erase(out_dims.begin() + axis);
    L->Reshape(out_dims);
    math::Copy(NxS, loss, L->template mutable_data<T, Context>(), ctx());
  } else {
    int64_t normalizer = 1;
    if (reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (reduction_ == "MEAN") {
      normalizer = NxS;
    }
    reduce_impl_.Setup<T>({NxS}, {0}, ctx());
    reduce_impl_.Compute<T>(
        loss,
        L->Reshape({})->template mutable_data<T, Context>(),
        ctx()->workspace()->template data<Context>(reduce_impl_.scratch_size()),
        ctx(),
        1.f / float(normalizer));
  }
}

template <class Context>
template <typename T>
void CNNLNLLLossGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), &dL = Input(2);
  auto* dX = Output(0)->ReshapeLike(X);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;

  auto* dl = dL.template data<T, Context>();
  auto* dx = dX->template mutable_data<T, Context>();
  auto* scratch = (T*)ctx()->workspace()->template data<Context>(
      sizeof(T) + NxS * sizeof(int));
  auto* target = Y.template data<int, Context>();
  auto* mutable_target = const_cast<int*>(target);
  if (this->ignore_index_ != INT_MAX) {
    mutable_target = reinterpret_cast<int*>(scratch + 1);
    kernels::Clip(NxS, 0.f, float(C - 1), target, mutable_target, ctx());
  }

  math::Set(dX->count(), convert::To<T>(0.f), dx, ctx());
  CNNLSetTensorDesc<T>(this->input_desc_, {1, 1, 1});
  CNNLSetTensorDesc<int>(this->target_desc_, {N, 1, S});
  CNNLSetTensorDesc<T>(this->output_desc_, {N, C, S});
  math::Set(1, convert::To<T>(-1.f), scratch, ctx());
  CNNL_CHECK(cnnlScatter(
      ctx()->cnnl_handle(),
      1,
      this->output_desc_,
      dx,
      this->target_desc_,
      mutable_target,
      this->input_desc_,
      scratch,
      this->output_desc_,
      dx,
      CNNL_SCATTER));
  kernels::MaskLossGrad(N, C, this->ignore_index_, target, dx, ctx());

  if (this->reduction_ == "NONE") {
    math::Mul(
        3,
        vec64_t({N, C, S}).data(),
        3,
        vec64_t({N, 1, S}).data(),
        dx,
        dl,
        dx,
        ctx());
  } else {
    int64_t normalizer = 1;
    if (this->reduction_ == "BATCH_MEAN") {
      normalizer = X.dim(0);
    } else if (this->reduction_ == "MEAN") {
      normalizer = NxS;
    }
    math::Scale(1, 1.f / float(normalizer), dl, scratch, ctx());
    math::Mul(
        3,
        vec64_t({N, C, S}).data(),
        1,
        vec64_t({1}).data(),
        dx,
        scratch,
        dx,
        ctx());
  }
}

DEPLOY_CNNL_OPERATOR(NLLLoss);
DEPLOY_CNNL_OPERATOR(NLLLossGradient);

} // namespace dragon

#endif // USE_MLU
