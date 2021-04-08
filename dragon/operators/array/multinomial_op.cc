#include "dragon/operators/array/multinomial_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MultinomialOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  auto* X_prob = Buffer("X_prob")->ReshapeLike(X);

  const int axis = X.ndim() - 1;
  const auto N = X.count(0, axis), C = X.dim(axis);
  vec64_t Y_dims(X.dims());
  Y_dims[axis] = sample_size_;

  CPUContext context;
  auto* p = X_prob->template mutable_data<T, CPUContext>();
  auto* y = Y->Reshape(Y_dims)->template mutable_data<int64_t, CPUContext>();
  kernels::Softmax(N, 1, C, X.template data<T, CPUContext>(), p, &context);

  vector<double> cumsum(C);
  auto* cdf = static_cast<double*>(cumsum.data());
  auto* rng = ctx()->rand_generator();

  int64_t index = 0;
  for (int i = 0; i < N; ++i) {
    double running_total = 0.;
    for (int j = 0; j < C; ++j) {
      running_total += double(p[j]);
      cdf[j] = running_total;
    }
    std::uniform_real_distribution<double> dist(0., running_total);
    for (int j = 0; j < sample_size_; ++j) {
      auto r = dist(*rng);
      auto found_iter = std::upper_bound(cdf, cdf + C, r);
      y[index++] = std::min(int64_t(std::distance(cdf, found_iter)), C - 1);
    }
    p += C;
  }
  Y->template data<int64_t, Context>();
}

template <class Context>
void MultinomialOp<Context>::RunOnDevice() {
  ctx()->set_stream(0); // Enforce the default stream
  DispatchHelper<dtypes::TypesBase<float, double>>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(Multinomial);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Multinomial);
#endif

OPERATOR_SCHEMA(Multinomial)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(Multinomial);

} // namespace dragon
