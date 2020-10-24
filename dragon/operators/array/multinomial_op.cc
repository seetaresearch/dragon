#include "dragon/operators/array/multinomial_op.h"
#include "dragon/core/workspace.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void MultinomialOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int axis = Input(0).ndim() - 1;
  vec64_t Y_dims(X.dims());
  Y_dims[axis] = num_samples_;

  double running_total, r;
  int yi = 0, num_classes = X.dim(axis);
  double uniform_p = 1. / (double)num_classes;

  vector<double> cumsum(num_classes);
  auto* cdf = static_cast<double*>(cumsum.data());
  auto* x = X.template data<T, CPUContext>();
  auto* y = Y->Reshape(Y_dims)->template mutable_data<int64_t, CPUContext>();

  if (normalize_) {
    CPUContext cpu_ctx;
    auto* prob = Buffer("prob")->template mutable_data<T, CPUContext>();
    kernel::Softmax(
        X.count(0, axis), X.count(axis + 1), X.dim(axis), x, prob, &cpu_ctx);
    x = prob;
  }

  auto* rng = ctx()->rand_generator();
  std::uniform_real_distribution<double> epsilon_dist;

  for (int i = 0; i < X.count(0, axis); ++i) {
    running_total = 0.;
    if (epsilon_ > 0. && epsilon_dist(*rng) < epsilon_) {
      for (int j = 0; j < num_classes; ++j) {
        running_total += uniform_p;
        cdf[j] = running_total;
      }
    } else {
      for (int j = 0; j < num_classes; ++j) {
        running_total += (double)x[j];
        cdf[j] = running_total;
      }
    }
    std::uniform_real_distribution<double> dist(0., running_total);
    for (int j = 0; j < (int)num_samples_; ++j) {
      r = dist(*rng);
      auto found_iter = std::upper_bound(cdf, cdf + num_classes, r);
      y[yi++] = std::min((int)std::distance(cdf, found_iter), num_classes - 1);
    }
    x += num_classes;
  }

  Y->template data<int64_t, Context>();
}

template <class Context>
void MultinomialOp<Context>::RunOnDevice() {
  ctx()->set_stream(0); // Enforce the default stream
  DispatchHelper<TensorTypes<float, double>>::Call(this, Input(0));
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
