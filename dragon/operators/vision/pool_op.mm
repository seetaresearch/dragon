#include "dragon/operators/vision/pool_op.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSPoolOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);

  vector<MPSGraphTensor_t> placeholders;
  if (mode_ == "MAX") {
    placeholders = this->graph_cache_.GetPlaceholders(
        {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
          this->SetPoolDesc();
          placeholders.emplace_back(MPSCreateTensor<T>(this->graph_, X.dims()));
          if (num_axes_ == 1 || num_axes_ == 2) {
            placeholders.emplace_back([this->graph_
                maxPooling2DWithSourceTensor:placeholders[0]
                                  descriptor:this->pool2d_desc_
                                        name:nil]);
          } else {
            LOG(FATAL) << "MaxPool" << num_axes_ << "d is not supported.";
          }
        });
  } else if (mode_ == "AVG") {
    placeholders = this->graph_cache_.GetPlaceholders(
        {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
          this->SetPoolDesc();
          placeholders.emplace_back(MPSCreateTensor<T>(this->graph_, X.dims()));
          if (num_axes_ == 1 || num_axes_ == 2) {
            placeholders.emplace_back([this->graph_
                avgPooling2DWithSourceTensor:placeholders[0]
                                  descriptor:this->pool2d_desc_
                                        name:nil]);
          } else {
            LOG(FATAL) << "AvgPool" << num_axes_ << "d is not supported.";
          }
        });
  }

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y->Reshape(out_shape_)->template mutable_data<T, Context>(),
          placeholders[1]),
    };
    ctx()->mps_stream()->Encode(this->graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSPoolGradientOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), &dY = Input(2), *dX = Output(0);

  vector<MPSGraphTensor_t> placeholders;
  if (mode_ == "MAX") {
    placeholders = this->graph_cache_.GetPlaceholders(
        {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
          this->SetPoolDesc();
          placeholders.emplace_back(MPSCreateTensor<T>(this->graph_, X.dims()));
          placeholders.push_back(MPSCreateTensor<T>(this->graph_, dY.dims()));
          if (num_axes_ == 1 || num_axes_ == 2) {
            placeholders.emplace_back([this->graph_
                maxPooling2DGradientWithGradientTensor:placeholders[1]
                                          sourceTensor:placeholders[0]
                                            descriptor:this->pool2d_desc_
                                                  name:nil]);
          } else {
            LOG(FATAL) << "MaxPool" << num_axes_ << "d is not supported.";
          }
        });
  } else if (mode_ == "AVG") {
    placeholders = this->graph_cache_.GetPlaceholders(
        {&X.dims()}, {&X.meta()}, [&](vector<MPSGraphTensor_t>& placeholders) {
          this->SetPoolDesc();
          placeholders.emplace_back(MPSCreateTensor<T>(this->graph_, X.dims()));
          placeholders.push_back(MPSCreateTensor<T>(this->graph_, dY.dims()));
          if (num_axes_ == 1 || num_axes_ == 2) {
            placeholders.emplace_back([this->graph_
                avgPooling2DGradientWithGradientTensor:placeholders[1]
                                          sourceTensor:placeholders[0]
                                            descriptor:this->pool2d_desc_
                                                  name:nil]);
          } else {
            LOG(FATAL) << "AvgPool" << num_axes_ << "d is not supported.";
          }
        });
  }

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
      placeholders[1] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[1]),
    };
    auto* outputs = @{
      placeholders[2] : MPSCreateTensorData(
          dX->ReshapeLike(X)->template mutable_data<T, Context>(),
          placeholders[2]),
    };
    ctx()->mps_stream()->Encode(this->graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(Pool, MPSPool);
DEPLOY_MPS_OPERATOR(PoolGradient, MPSPoolGradient);

} // namespace dragon
