#ifdef USE_MLU

#include "dragon/core/workspace.h"
#include "dragon/operators/normalization/lp_norm_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void CNNLLpNormOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);

  const auto N = X.count(0, axis), S = X.count(end_axis + 1);
  const auto C = X.count(axis, end_axis + 1);
  CNNLSetTensorDesc<T>(input_desc_, {N, C, S});
  CNNLSetTensorDesc<T>(stats_desc_, {N, 1, S});

  int norm_axis = 1;
  size_t scratch_size = 0;
  CNNL_CHECK(cnnlSetNormalizeDescriptor_v2(
      norm_desc_,
      &norm_axis,
      1,
      CNNL_NOT_PROPAGATE_NAN,
      epsilon_,
      float(p_),
      0, // channel_shared
      0)); // across_spatial
  CNNL_CHECK(cnnlGetNormalizeWorkspaceSize(
      ctx()->cnnl_handle(),
      norm_desc_,
      input_desc_,
      input_desc_,
      stats_desc_,
      &scratch_size));
  CNNL_CHECK(cnnlNormalize_v3(
      ctx()->cnnl_handle(),
      norm_desc_,
      input_desc_,
      X.template data<T, Context>(),
      nullptr,
      nullptr,
      ctx()->workspace()->template data<Context>(scratch_size),
      scratch_size,
      input_desc_,
      Y->ReshapeLike(X)->template mutable_data<T, Context>(),
      stats_desc_,
      Output("X_stats")->Reshape({N, S})->template mutable_data<T, Context>()));
}

template <class Context>
template <typename T>
void CNNLLpNormGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);
  GET_OP_AXIS_ARG(end_axis, X.ndim(), axis);
  const auto N = X.count(0, axis), S = X.count(end_axis + 1);
  const auto C = X.count(axis, end_axis + 1), NxS = N * S;
  const vec64_t X_dims({N, C, S}), stats_dims({N, 1, S});

  auto* x = X.template data<T, Context>();
  auto* dy = dY.template data<T, Context>();
  auto* dx = dX->ReshapeLike(X)->template mutable_data<T, Context>();
  auto* norm = Input("X_stats").template mutable_data<T, Context>();
  auto* data1 = ctx()->workspace()->template data<T, Context>(
      NxS + X.count(), "BufferKernel");
  auto* data2 = data1 + NxS;

  math::Inv(NxS, norm, norm, ctx());
  math::Square(NxS, norm, data1, ctx());
  math::Mul(X.count(), dy, x, dx, ctx());
  reduce_impl_.Setup<T>({N, C, S}, {1}, ctx());
  reduce_impl_.Compute<T>(
      dx,
      data2,
      ctx()->workspace()->template data<Context>(reduce_impl_.scratch_size()),
      ctx());
  math::Mul(NxS, data1, data2, data1, ctx());
  if (p_ == 1) {
    math::Sign(X.count(), x, data2, ctx());
    math::Mul(3, &X_dims[0], 3, &stats_dims[0], data2, data1, data2, ctx());
  } else if (p_ == 2) {
    math::Mul(NxS, data1, norm, data1, ctx());
    math::Mul(3, &X_dims[0], 3, &stats_dims[0], x, data1, data2, ctx());
  } else {
    LOG(FATAL) << "Unsupported order of LpNormalization: " << p_;
  }
  math::Mul(3, &X_dims[0], 3, &stats_dims[0], dy, norm, dx, ctx());
  math::Sub(X.count(), dx, data2, dx, ctx());
}

DEPLOY_CNNL_OPERATOR(LpNorm);
DEPLOY_CNNL_OPERATOR(LpNormGradient);

} // namespace dragon

#endif // USE_MLU
