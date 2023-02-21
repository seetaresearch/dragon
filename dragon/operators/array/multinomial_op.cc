#include "dragon/operators/array/multinomial_op.h"
#include "dragon/kernels/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MultinomialOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto* X_prob = Output("X_prob")->ReshapeLike(X);

  const int axis = X.ndim() - 1;
  const auto N = X.count(0, axis), C = X.dim(axis);
  vec64_t Y_dims(X.dims());
  Y_dims[axis] = sample_size_;

  auto* x = X_prob->template mutable_data<T, CPUContext>();
  auto* y = Y->Reshape(Y_dims)->template mutable_data<int64_t, CPUContext>();
  kernels::Softmax(N, 1, C, X.template data<T, CPUContext>(), x, &impl_ctx_);

  vector<double> cumsum(C);
  auto* cdf = static_cast<double*>(cumsum.data());
  auto* rng = impl_ctx_.rand_generator();

  int64_t index = 0;
  for (int i = 0; i < N; ++i, x += C) {
    double running_total = 0.;
    for (int j = 0; j < C; ++j) {
      running_total += double(x[j]);
      cdf[j] = running_total;
    }
    std::uniform_real_distribution<double> dist(0., running_total);
    for (int j = 0; j < sample_size_; ++j) {
      auto found_iter = std::upper_bound(cdf, cdf + C, dist(*rng));
      y[index++] = std::min(int64_t(std::distance(cdf, found_iter)), C - 1);
    }
  }
  Y->template data<int64_t, Context>();
}

DEPLOY_CPU_OPERATOR(Multinomial);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Multinomial);
#endif
#ifdef USE_MPS
DEPLOY_MPS_OPERATOR(Multinomial, Multinomial);
#endif
#ifdef USE_MLU
DEPLOY_MLU_OPERATOR(Multinomial);
#endif

OPERATOR_SCHEMA(Multinomial).NumInputs(1).NumOutputs(1);

NO_GRADIENT(Multinomial);

} // namespace dragon
