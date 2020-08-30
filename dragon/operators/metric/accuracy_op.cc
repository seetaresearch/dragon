#include "dragon/operators/metric/accuracy_op.h"

namespace dragon {

template <class Context>
template <typename LogitType, typename TargetType>
void AccuracyOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);
  CANONICALIZE_AXIS_WITH_TENSOR(X);

  auto outer_dim = X.count(0, axis);
  auto axis_dim = X.dim(axis);
  auto inner_dim = X.count(axis + 1);

  CHECK_EQ(outer_dim * inner_dim, Input(1).count())
      << "\nNumber of preds must match the number of targets.";

  int64_t acc = 0, count = 0;
  int64_t cols = X.count() / outer_dim;

  auto* logit = X.template data<LogitType, CPUContext>();
  auto* target = Input(1).template data<TargetType, CPUContext>();

  for (int i = 0; i < outer_dim; ++i) {
    for (int j = 0; j < inner_dim; ++j) {
      const int label = target[i * inner_dim + j];
      if (label == ignore_index_) continue;
      vector<pair<LogitType, int>> vec;
      for (int k = 0; k < axis_dim; k++)
        vec.push_back(std::make_pair(logit[i * cols + k * inner_dim + j], k));
      std::partial_sort(
          vec.begin(),
          vec.begin() + top_k_,
          vec.end(),
          std::greater<pair<LogitType, int>>());
      for (int k = 0; k < top_k_; k++) {
        if (vec[k].second == label) {
          acc++;
          break;
        }
      }
      count++;
    } // End inner_dim
  } // End outer_dim

  Y->Reshape({})->template mutable_data<float, CPUContext>()[0] =
      (float)acc / (float)count;
}

template <class Context>
void AccuracyOp<Context>::RunOnDevice() {
  if (Input(0).template IsType<float>()) {
    if (Input(1).template IsType<float>()) {
      DoRunWithType<float, float>();
    } else if (Input(1).template IsType<int64_t>()) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float32", "int64"});
    }
  } else if (Input(0).template IsType<double>()) {
    if (Input(1).template IsType<double>()) {
      DoRunWithType<double, double>();
    } else if (Input(1).template IsType<int64_t>()) {
      DoRunWithType<double, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          types::to_string(Input(1).meta()), {"float64", "int64"});
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        types::to_string(Input(0).meta()), {"float32", "float64"});
  }
}

DEPLOY_CPU_OPERATOR(Accuracy);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Accuracy);
#endif

OPERATOR_SCHEMA(Accuracy)
    /* X, T */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(Accuracy);

} // namespace dragon
