#include "dragon/operators/vision/conv_transpose_op.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSConvTransposeOp<Context>::DoRunWithType() {
  ConvOpBase<Context>::Reshape();
  auto &X = Input(0), &W = Input(1), *Y = Output(0);
  INITIALIZE_TENSOR_VIA_SPEC(W, w_shape_, T);
  if (HasBias()) {
    INITIALIZE_TENSOR_VIA_SPEC(Input(2), b_shape_, T);
  }

  vec64_t X_dims(X.dims()), W_dims(W.dims()), B_dims({1, out_channels_});
  vec64_t Y_dims(Y->dims());
  for (int i = 0; i < std::max(num_axes_, int64_t(2)); ++i) {
    if (W_dims.size() < 4) W_dims.push_back(1);
    if (data_format() == "NCHW") {
      if (X_dims.size() < 4) X_dims.push_back(1);
      if (Y_dims.size() < 4) Y_dims.push_back(1);
      B_dims.push_back(1);
    } else {
      if (X_dims.size() < 4) X_dims.insert(X_dims.end() - 1, 1);
      if (Y_dims.size() < 4) Y_dims.insert(Y_dims.end() - 1, 1);
      B_dims.insert(B_dims.begin() + 1, 1);
    }
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &W.dims()},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, W_dims));
        // clang-format off
        if (num_axes_ == 1 || num_axes_ == 2) {
          placeholders.emplace_back([graph_
              convolution2DDataGradientWithIncomingGradientTensor:placeholders[0]
                                                    weightsTensor:placeholders[1]
                                                      outputShape:MPSGetShape(Y_dims)
                                     forwardConvolutionDescriptor:conv2d_desc_
                                                             name:nil]);
        } else {
          LOG(FATAL) << "Conv" << num_axes_ << "d is not supported.";
        }
        // clang-format off
        if (HasBias()) {
          placeholders.emplace_back(MPSCreateTensor<T>(graph_, B_dims));
          placeholders[2] = [graph_ additionWithPrimaryTensor:placeholders[2]
                                              secondaryTensor:placeholders[3]
                                                         name:nil];
        }
      });

  @autoreleasepool {
    auto* inputs = [[[NSMutableDictionary alloc] init] autorelease];
    inputs[placeholders[0]] = MPSCreateTensorData(
        X.template data<T, Context>(), placeholders[0]);
    inputs[placeholders[1]] = MPSCreateTensorData(
        W.template data<T, Context>(), placeholders[1]);
    if (HasBias()) {
      inputs[placeholders[3]] = MPSCreateTensorData(
          Input(2).template data<T, Context>(), placeholders[3]);
    }
    auto* outputs = @{
      placeholders[2] : MPSCreateTensorData(
          Y->template mutable_data<T, Context>(), placeholders[2]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSConvTransposeGradientOp<Context>::DoRunWithType() {
  ConvOpBase<Context>::Reshape(true);
  auto &X = Input(0), &W = Input(1), &dY = Input(-1);
  auto *dX = Output(0), *dW = Output(1);

  vec64_t X_dims(X.dims()), W_dims(W.dims()), dY_dims(dY.dims());
  if (num_axes_ == 1) {
    W_dims.push_back(1);
    if (data_format() == "NCHW") {
      X_dims.push_back(1);
      dY_dims.push_back(1);
    } else {
      X_dims.insert(X_dims.end() - 1, 1);
      dY_dims.insert(dY_dims.end() - 1, 1);
    }
  }

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &W.dims()},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, W_dims));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY_dims));
        // clang-format off
        if (num_axes_ == 1 || num_axes_ == 2) {
          placeholders.emplace_back([graph_
              convolution2DWithSourceTensor:placeholders[2]
                              weightsTensor:placeholders[1]
                                 descriptor:conv2d_desc_
                                       name:nil]);
          placeholders.emplace_back([graph_
              convolution2DWeightsGradientWithIncomingGradientTensor:placeholders[0]
                                                        sourceTensor:placeholders[2]
                                                         outputShape:MPSGetShape(W_dims)
                                        forwardConvolutionDescriptor:conv2d_desc_
                                                                name:nil]);
        } else {
          LOG(FATAL) << "Conv" << num_axes_ << "d is not supported.";
        }
        // clang-format on
        vec64_t B_axes = {0};
        for (int i = 0; i < num_axes_; ++i) {
          B_axes.push_back(axis_ + i);
        }
        placeholders.emplace_back([graph_
            reductionSumWithTensor:placeholders[2]
                              axes:MPSGetShape(B_axes)
                              name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
      placeholders[1] :
          MPSCreateTensorData(W.template data<T, Context>(), placeholders[1]),
      placeholders[2] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[2]),
    };
    auto* outputs = [[[NSMutableDictionary alloc] init] autorelease];
    if (dX->has_name()) {
      outputs[placeholders[3]] = MPSCreateTensorData(
          dX->template mutable_data<T, Context>(), placeholders[3]);
    }
    if (dW->has_name()) {
      outputs[placeholders[4]] = MPSCreateTensorData(
          dW->template mutable_data<T, Context>(), placeholders[4]);
    }
    if (HasBias()) {
      outputs[placeholders[5]] = MPSCreateTensorData(
          Output(2)->template mutable_data<T, Context>(), placeholders[5]);
    }
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(ConvTranspose, MPSConvTranspose);
DEPLOY_MPS_OPERATOR(ConvTransposeGradient, MPSConvTransposeGradient);

} // namespace dragon
