#include "dragon/operators/math/matmul_op.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSMatMulOp<Context>::DoRunGraphWithType(
    const vector<MPSGraphTensor_t>& placeholders,
    const vec64_t& Y_dims) {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(A.template data<T, Context>(), placeholders[0]),
      placeholders[1] :
          MPSCreateTensorData(B.template data<T, Context>(), placeholders[1]),
    };
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
void MPSMatMulOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), *Y = Output(0);
  auto A_ndim = A.ndim(), B_ndim = B.ndim();

  if (A_ndim == 1 && B_ndim == 1) {
    // Vector @ Vector.
    CHECK_EQ(A.count(), B.count()) << "\nExcept equal length of two vectors.";
    auto placeholders = graph_cache_.GetPlaceholders(
        {&A.dims(), &B.dims()},
        {&A.meta()},
        [&](vector<MPSGraphTensor_t>& placeholders) {
          placeholders.emplace_back(MPSCreateTensor<T>(graph_, A.dims()));
          placeholders.emplace_back(MPSCreateTensor<T>(graph_, B.dims()));
          auto* AxB = [graph_ multiplicationWithPrimaryTensor:placeholders[0]
                                              secondaryTensor:placeholders[1]
                                                         name:nil];
          placeholders.emplace_back([graph_ reductionSumWithTensor:AxB
                                                              axes:nil
                                                              name:nil]);
        });
    return this->DoRunGraphWithType<T>(placeholders, {});
  }

  if (A_ndim == 1) {
    NOT_IMPLEMENTED;
  }

  if (B_ndim == 1) {
    // Matrix @ Vector.
    const auto N = B.count();
    CHECK_EQ(A.dim(A_ndim - 1), N) << "\nExcept the last dim of A is " << N
                                   << ", got " << A.dim(A_ndim - 1);
    vec64_t Y_dims(A.dims()), B_dims(B.dims());
    Y_dims.erase(Y_dims.end() - 1);
    B_dims.push_back(1);
    auto placeholders = graph_cache_.GetPlaceholders(
        {&A.dims(), &B.dims()},
        {&A.meta()},
        [&](vector<MPSGraphTensor_t>& placeholders) {
          placeholders.emplace_back(MPSCreateTensor<T>(graph_, A.dims()));
          placeholders.emplace_back(MPSCreateTensor<T>(graph_, B_dims));
          placeholders.emplace_back([graph_
              matrixMultiplicationWithPrimaryTensor:placeholders[0]
                                    secondaryTensor:placeholders[1]
                                               name:nil]);
        });
    return this->DoRunGraphWithType<T>(placeholders, Y_dims);
  }

  // Check matrix A.
  const auto M = A.dim(A_ndim - 2);
  const auto K = A.dim(A_ndim - 1);

  // Check matrix B.
  CHECK_EQ(B.dim(B_ndim - 2), K) << "\nExcept the second last dim of B is " << K
                                 << ", got " << B.dim(B_ndim - 2);
  const auto N = B.dim(B_ndim - 1);

  // Check batching && broadcasting.
  vec64_t A_dims(A.dims().begin(), A.dims().end() - 2);
  vec64_t B_dims(B.dims().begin(), B.dims().end() - 2), Y_dims;
  CHECK(math::utils::IsBinaryBroadcast(A_dims, B_dims, Y_dims))
      << "\nCould not broadcast with " << A.DimString() << " " << B.DimString();
  Y_dims.push_back(M);
  Y_dims.push_back(N);

  auto placeholders = graph_cache_.GetPlaceholders(
      {&A.dims(), &B.dims()},
      {&A.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, A.dims()));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, B.dims()));
        placeholders.emplace_back([graph_
            matrixMultiplicationWithPrimaryTensor:placeholders[0]
                                  secondaryTensor:placeholders[1]
                                             name:nil]);
      });
  this->DoRunGraphWithType<T>(placeholders, Y_dims);
}

template <class Context>
template <typename T>
void MPSMatMulGradientOp<Context>::DoRunWithType() {
  auto &A = Input(0), &B = Input(1), &dY = Input(2);
  auto *dA = Output(0), *dB = Output(1);
  auto A_ndim = A.ndim(), B_ndim = B.ndim();

  if (A_ndim == 1 && B_ndim == 1) {
    NOT_IMPLEMENTED;
  }

  if (A_ndim == 1) {
    NOT_IMPLEMENTED;
  }

  if (B_ndim == 1) {
    NOT_IMPLEMENTED;
  }

  // Check matrix A && B.
  const auto M = A.dim(A_ndim - 2);
  const auto K = A.dim(A_ndim - 1);
  const auto N = B.dim(B_ndim - 1);

  // Check batching && broadcasting.
  vec64_t A_dims(A.dims().begin(), A.dims().end() - 2);
  vec64_t B_dims(B.dims().begin(), B.dims().end() - 2);
  vec64_t A_batch_dims, B_batch_dims, Y_batch_dims;
  vec64_t A_batch_axes, B_batch_axes;
  CHECK(math::utils::IsBinaryBroadcast(A_dims, B_dims, Y_batch_dims))
      << "\nCould not broadcast with " << A.DimString() << " " << B.DimString();
  math::utils::ComputeBroadcastDims(A_dims, B_dims, A_batch_dims, B_batch_dims);
  math::utils::ComputeBroadcastAxes(
      A_batch_dims, B_batch_dims, Y_batch_dims, A_batch_axes, B_batch_axes);
  A_dims = A_batch_dims, B_dims = B_batch_dims;
  A_dims.push_back(M);
  A_dims.push_back(K);
  B_dims.push_back(K);
  B_dims.push_back(N);

  auto placeholders = graph_cache_.GetPlaceholders(
      {&A.dims(), &B.dims()},
      {&A.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, A_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, B_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY.dims()));
        auto* A_transpose = [graph_ transposeTensor:placeholders[0]
                                          dimension:-1
                                      withDimension:-2
                                               name:nil];
        auto* B_transpose = [graph_ transposeTensor:placeholders[1]
                                          dimension:-1
                                      withDimension:-2
                                               name:nil];
        placeholders.emplace_back([graph_
            matrixMultiplicationWithPrimaryTensor:placeholders[2]
                                  secondaryTensor:B_transpose
                                             name:nil]);
        placeholders.emplace_back([graph_
            matrixMultiplicationWithPrimaryTensor:A_transpose
                                  secondaryTensor:placeholders[2]
                                             name:nil]);
        placeholders[3] = A_batch_axes.empty()
            ? placeholders[3]
            : [graph_ reductionSumWithTensor:placeholders[3]
                                        axes:MPSGetShape(A_batch_axes)
                                        name:nil];
        placeholders[4] = B_batch_axes.empty()
            ? placeholders[4]
            : [graph_ reductionSumWithTensor:placeholders[4]
                                        axes:MPSGetShape(B_batch_axes)
                                        name:nil];
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
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(MatMul, MPSMatMul);
DEPLOY_MPS_OPERATOR(MatMulGradient, MPSMatMulGradient);

} // namespace dragon
