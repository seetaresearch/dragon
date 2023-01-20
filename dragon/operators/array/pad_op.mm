#include "dragon/operators/array/pad_op.h"
#include "dragon/core/workspace.h"

namespace dragon {

template <class Context>
template <typename T>
void MPSPadOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0);

  int num_pads, num_dims = X.ndim();
  pads(0, &num_pads);

  CHECK_EQ(num_pads, num_dims * 2)
      << "\nGot " << num_pads << " pads (" << num_pads / 2 << " axes) "
      << "for input dimensions " << X.DimString() << ".";

  vec64_t X_pads(num_dims * 2), Y_dims(X.dims());
  for (int i = 0; i < num_dims; ++i) {
    X_pads[i] = pads(i);
    X_pads[i + num_dims] = pads(i + num_dims);
    Y_dims[i] += (X_pads[i] + X_pads[i + num_dims]);
  }

  // Save for the gradient computation.
  Output("X_pads")->template CopyFrom<int64_t>(X_pads);

  if (X.dims() == Y_dims) {
    Y->Reshape(Y_dims)->CopyFrom(X, ctx());
    return;
  }

  vec64_t X_pads_begin({X_pads.begin(), X_pads.begin() + num_dims});
  vec64_t X_pads_end({X_pads.begin() + num_dims, X_pads.end()});

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &X_pads},
      {&X.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        MPSGraphPaddingMode mode;
        if (mode_ == "CONSTANT") {
          mode = MPSGraphPaddingModeConstant;
        } else if (mode_ == "REFLECT") {
          mode = MPSGraphPaddingModeReflect;
        } else if (mode_ == "EDGE") {
          mode = MPSGraphPaddingModeClampToEdge;
        } else {
          LOG(FATAL) << "Unsupported Pad mode: " << mode_;
        }
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        placeholders.emplace_back([graph_ padTensor:placeholders[0]
                                    withPaddingMode:mode
                                        leftPadding:MPSGetShape(X_pads_begin)
                                       rightPadding:MPSGetShape(X_pads_end)
                                      constantValue:value_
                                               name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
    };
    auto* outputs = @{
      placeholders[1] : MPSCreateTensorData(
          Y->Reshape(Y_dims)->template mutable_data<T, Context>(),
          placeholders[1]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

template <class Context>
template <typename T>
void MPSPadGradientOp<Context>::DoRunWithType() {
  auto &X = Input(0), &dY = Input(1), *dX = Output(0);

  if (dY.dims() == X.dims()) {
    dX->Reshape(X.dims())->CopyFrom(dY, ctx());
    return; // Just copy the contents
  }

  // Restore the input dimensions
  int num_dims = dY.ndim();
  vec64_t X_dims(dY.dims()), X_pads;
  Input("X_pads").template CopyTo<int64_t>(X_pads);
  for (int i = 0; i < num_dims; ++i) {
    X_dims[i] -= (X_pads[i] + X_pads[i + num_dims]);
  }

  vec64_t X_pads_begin({X_pads.begin(), X_pads.begin() + num_dims});
  vec64_t X_pads_end({X_pads.begin() + num_dims, X_pads.end()});

  auto placeholders = graph_cache_.GetPlaceholders(
      {&X.dims(), &X_pads},
      {&dY.meta()},
      [&](vector<MPSGraphTensor_t>& placeholders) {
        MPSGraphPaddingMode mode;
        if (mode_ == "CONSTANT") {
          mode = MPSGraphPaddingModeConstant;
        } else if (mode_ == "REFLECT") {
          mode = MPSGraphPaddingModeReflect;
        } else if (mode_ == "EDGE") {
          mode = MPSGraphPaddingModeClampToEdge;
        } else {
          LOG(FATAL) << "Unsupported Pad mode: " << mode_;
        }
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, X.dims()));
        placeholders.emplace_back(MPSCreateTensor<T>(graph_, dY.dims()));
        placeholders.emplace_back([graph_
            padGradientWithIncomingGradientTensor:placeholders[1]
                                     sourceTensor:placeholders[0]
                                      paddingMode:mode
                                      leftPadding:MPSGetShape(X_pads_begin)
                                     rightPadding:MPSGetShape(X_pads_end)
                                             name:nil]);
      });

  @autoreleasepool {
    auto* inputs = @{
      placeholders[0] :
          MPSCreateTensorData(X.template data<T, Context>(), placeholders[0]),
      placeholders[1] :
          MPSCreateTensorData(dY.template data<T, Context>(), placeholders[1]),
    };
    auto* outputs = @{
      placeholders[2] : MPSCreateTensorData(
          dX->Reshape(X.dims())->template mutable_data<T, Context>(),
          placeholders[2]),
    };
    ctx()->mps_stream()->Encode(graph_, inputs, outputs);
  }
}

DEPLOY_MPS_OPERATOR(Pad, MPSPad);
DEPLOY_MPS_OPERATOR(PadGradient, MPSPadGradient);

DEFINE_OP_REPEATED_ARG(int64_t, MPSPadOp, pads);

} // namespace dragon
