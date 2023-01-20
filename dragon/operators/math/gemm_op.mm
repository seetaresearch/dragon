#include "dragon/operators/math/gemm_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSGemmOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  const auto A_axis = A.axis(-1), B_axis = B.axis(-1);

  // Check matrix A.
  auto M = transA_ ? A.count(A_axis) : A.count(0, A_axis);
  auto K = transA_ ? A.count(0, A_axis) : A.count(A_axis);

  // Check matrix B.
  auto N = transB_ ? B.count(0, B_axis) : B.count(B_axis);
  auto K2 = transB_ ? B.count(B_axis) : B.count(0, B_axis);
  if (transB_) {
    CHECK_EQ(K, K2) << "\nMatrixB's dimensions should be (...," << K
                    << "), got " << B.DimString() << ".";
  } else {
    CHECK_EQ(K, K2) << "\nMatrixB's dimensions should be reshaped to (" << K
                    << "," << N << "), got " << B.DimString() << ".";
  }

  vec64_t Y_dims, num_inputs = {InputSize()};
  if (transA_) {
    Y_dims.push_back(M);
  } else {
    Y_dims.insert(Y_dims.end(), A.dims().begin(), A.dims().begin() + A_axis);
  }
  if (transB_) {
    Y_dims.insert(Y_dims.end(), B.dims().begin(), B.dims().begin() + B_axis);
  } else {
    Y_dims.push_back(N);
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&A.dims(), &B.dims(), &num_inputs},
      {&A.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, A.dims()));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, B.dims()));
        auto* A_mat = transA_ ? [graph_ transposeTensor:placeholders[0]
                                              dimension:-1
                                          withDimension:-2
                                                   name:nil]
                              : placeholders[0];
        auto* B_mat = transB_ ? [graph_ transposeTensor:placeholders[1]
                                              dimension:-1
                                          withDimension:-2
                                                   name:nil]
                              : placeholders[1];
        placeholders.emplace_back([graph_
            matrixMultiplicationWithPrimaryTensor:A_mat
                                  secondaryTensor:B_mat
                                             name:nil]);
        if (InputSize() > 2) {
          placeholders.emplace_back(
              MPSCreateTensor<T>(graph_, Input(2).dims()));
          placeholders[2] = [graph_ additionWithPrimaryTensor:placeholders[2]
                                              secondaryTensor:placeholders[3]
                                                         name:nil];
        }
      });

  @autoreleasepool {
    auto* inputs = [[[NSMutableDictionary alloc] init] autorelease];
    inputs[placeholders[0]] =
        MPSCreateTensorData(A.template data<T, Context>(), placeholders[0]);
    inputs[placeholders[1]] =
        MPSCreateTensorData(B.template data<T, Context>(), placeholders[1]);
    if (placeholders.size() > 3) {
      inputs[placeholders[3]] = MPSCreateTensorData(
          Input(2).template data<T, Context>(), placeholders[3]);
    }
    auto* outputs = @{
      placeholders[2] : MPSCreateTensorData(
          Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
          placeholders[2]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSGemmGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &C = Input(2), &dY = Input(3);
  auto *dA = Output(0), *dB = Output(1), *dC = Output(2);
  const auto A_axis = A.axis(-1), B_axis = B.axis(-1);

  // Check matrix A/B.
  auto M = transA_ ? A.count(A_axis) : A.count(0, A_axis);
  auto N = transB_ ? B.count(0, B_axis) : B.count(B_axis);

  const auto A_dims = vec64_t({A.count(0, A_axis), A.count(A_axis)});
  const auto B_dims = vec64_t({B.count(0, B_axis), B.count(B_axis)});
  const auto dY_dims = vec64_t({M, N});

  auto placeholders = graph_cache_.GetPlaceholders(
      {&dY.dims(), &B.dims()},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, A_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, B_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY_dims));
        auto* A_transpose = [graph_ transposeTensor:placeholders[0]
                                          dimension:-1
                                      withDimension:-2
                                               name:nil];
        auto* B_transpose = [graph_ transposeTensor:placeholders[1]
                                          dimension:-1
                                      withDimension:-2
                                               name:nil];
        auto* dY_transpose = [graph_ transposeTensor:placeholders[2]
                                           dimension:-1
                                       withDimension:-2
                                                name:nil];
        if (transA_ > 0) {
          auto* B_mat = transB_ ? B_transpose : placeholders[1];
          placeholders.emplace_back([graph_
              matrixMultiplicationWithPrimaryTensor:B_mat
                                    secondaryTensor:dY_transpose
                                               name:nil]);
        } else {
          auto* B_mat = transB_ ? placeholders[1] : B_transpose;
          placeholders.emplace_back([graph_
              matrixMultiplicationWithPrimaryTensor:placeholders[2]
                                    secondaryTensor:B_mat
                                               name:nil]);
        }
        if (transB_ > 0) {
          auto* A_mat = transA_ ? A_transpose : placeholders[0];
          placeholders.emplace_back([graph_
              matrixMultiplicationWithPrimaryTensor:dY_transpose
                                    secondaryTensor:A_mat
                                               name:nil]);
        } else {
          auto* A_mat = transA_ ? placeholders[0] : A_transpose;
          placeholders.emplace_back([graph_
              matrixMultiplicationWithPrimaryTensor:A_mat
                                    secondaryTensor:placeholders[2]
                                               name:nil]);
        }
        placeholders.emplace_back([graph_
            reductionSumWithTensor:placeholders[2]
                              axes:MPSGetShape(vec64_t({0}))
                              name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(A.template data<T, Context>(), placeholders[0]),
      placeholders[1] :
          MPSCreateTensorData(B.template data<T, Context>(), placeholders[1]),
      placeholders[2] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[2]),
    };
    auto* outputs = [[[NSMutableDictionary alloc] init] autorelease];
    if (dA->has_name()) {
      outputs[placeholders[3]] = MPSCreateTensorData(
          dA->ReshapeLike(A)->template mutable_data<T, Context>(),
          placeholders[3]);
    }
    if (dB->has_name()) {
      outputs[placeholders[4]] = MPSCreateTensorData(
          dB->ReshapeLike(B)->template mutable_data<T, Context>(),
          placeholders[4]);
    }
    if (dC->has_name()) {
      CHECK_EQ(C.count(), N) << "\nUnsupported bias broadcasting.";
      outputs[placeholders[5]] = MPSCreateTensorData(
          dC->ReshapeLike(C)->template mutable_data<T, Context>(),
          placeholders[5]);
    }
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(Gemm, MPSGemm);
DEPLOY_MPS_OPERATOR(GemmGradient, MPSGemmGradient);

} // namespace dragon
