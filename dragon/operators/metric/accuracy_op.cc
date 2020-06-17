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
  if (XIsType(Input(0), float)) {
    if (XIsType(Input(1), float)) {
      DoRunWithType<float, float>();
    } else if (XIsType(Input(1), int64_t)) {
      DoRunWithType<float, int64_t>();
    } else {
      LOG(FATAL) << TypeString(Input(1), {"int64", "float32"});
    }
  } else if (XIsType(Input(0), double)) {
    if (XIsType(Input(1), double)) {
      DoRunWithType<double, double>();
    } else if (XIsType(Input(1), int64_t)) {
      DoRunWithType<double, int64_t>();
    } else {
      LOG(FATAL) << TypeString(Input(1), {"int64", "float64"});
    }
  } else {
    LOG(FATAL) << TypeString(Input(0), {"float32", "float64"});
  }
}

DEPLOY_CPU(Accuracy);
#ifdef USE_CUDA
DEPLOY_CUDA(Accuracy);
#endif

OPERATOR_SCHEMA(Accuracy)
    /* X, T */
    .NumInputs(2)
    /* Y */
    .NumOutputs(1);

NO_GRADIENT(Accuracy);

} // namespace dragon
