#include "dragon/operators/vision/resize_op.h"
#include "dragon/core/workspace.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSResizeOp<Context>::DoRunWithType() {
  ComputeOutShape();
  auto &X = Input(0), *Y = Output(0);
  Output("X_spec")->ReshapeLike(X);

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &out_dims_},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        placeholders.emplace_back([graph_
            resizeTensor:placeholders[0]
                    size:MPSGetShape(out_dims_)
                    mode:(mode_ == "NEAREST") ? MPSGraphResizeNearest
                                              : MPSGraphResizeBilinear
            centerResult:true
            alignCorners:align_corners_ > 0
                  layout:(data_format() == "NCHW")
                      ? MPSGraphTensorNamedDataLayoutNCHW
                      : MPSGraphTensorNamedDataLayoutNHWC
                    name:nil]);
      });

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
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSResizeGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0)->ReshapeLike(Input("X_spec"));

  auto placeholders = graph_cache_.GetPlaceholders(
      {&dX->dims(), &out_dims_},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        auto* X_dummy = [graph_ constantWithScalar:0
                                             shape:MPSGetShape(dX->dims())
                                          dataType:MPSGetDataType(dY.meta())];
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY.dims()));
        placeholders.emplace_back([graph_
            resizeWithGradientTensor:placeholders[0]
                               input:X_dummy
                                mode:(mode_ == "NEAREST")
                                    ? MPSGraphResizeNearest
                                    : MPSGraphResizeBilinear
                        centerResult:true
                        alignCorners:align_corners_ > 0
                              layout:(data_format() == "NCHW")
                                  ? MPSGraphTensorNamedDataLayoutNCHW
                                  : MPSGraphTensorNamedDataLayoutNHWC
                                name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          dX->template mutable_data<T, Context>(), placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(Resize, MPSResize);
DEPLOY_MPS_OPERATOR(ResizeGradient, MPSResizeGradient);

} // namespace dragon
