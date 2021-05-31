#include "dragon/operators/metric/accuracy_op.h"

namespace dragon {

template <class Context>
template <typename InputT, typename TargetT>
void AccuracyOp<Context>::DoRunWithType() {
  auto &X = Input(0), &Y = Input(1), *R = Output(0);
  GET_OP_AXIS_ARG(axis, X.ndim(), -1);

  const auto C = X.dim(axis);
  const auto N = X.count(0, axis);
  const auto S = X.count(axis + 1);
  const auto NxS = N * S;
  const auto CxS = C * S;
  CHECK_EQ(Y.count(), NxS) << "\nNumel of X and Y must be matched.";

  auto* input = X.template data<InputT, CPUContext>();
  auto* target = Y.template data<TargetT, CPUContext>();

  int64_t acc = 0, count = 0;
  for (int i = 0; i < N; ++i) {
    for (int j = 0; j < S; ++j) {
      const int label = target[i * S + j];
      if (label == ignore_index_) continue;
      vector<pair<InputT, int>> vec;
      for (int k = 0; k < C; ++k) {
        vec.push_back(std::make_pair(input[i * CxS + k * S + j], k));
      }
      std::partial_sort(
          vec.begin(),
          vec.begin() + top_k_,
          vec.end(),
          std::greater<pair<InputT, int>>());
      for (int k = 0; k < top_k_; k++) {
        if (vec[k].second == label) {
          acc++;
          break;
        }
      }
      count++;
    }
  }

  R->Reshape({})->template mutable_data<float, CPUContext>()[0] =
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
          dtypes::to_string(Input(1).meta()), {"float32", "int64"});
    }
  } else if (Input(0).template IsType<double>()) {
    if (Input(1).template IsType<double>()) {
      DoRunWithType<double, double>();
    } else if (Input(1).template IsType<int64_t>()) {
      DoRunWithType<double, int64_t>();
    } else {
      LOG(FATAL) << MessageForUnsupported(
          dtypes::to_string(Input(1).meta()), {"float64", "int64"});
    }
  } else {
    LOG(FATAL) << MessageForUnsupported(
        dtypes::to_string(Input(0).meta()), {"float32", "float64"});
  }
}

DEPLOY_CPU_OPERATOR(Accuracy);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(Accuracy);
#endif

OPERATOR_SCHEMA(Accuracy)
    /* X, Y */
    .NumInputs(2)
    /* R */
    .NumOutputs(1);

NO_GRADIENT(Accuracy);

} // namespace dragon
