#include "dragon/core/workspace.h"
#include "dragon/operators/activation/drop_block_op.h"
#include "dragon/utils/math_functions.h"
#include "dragon/utils/op_kernels.h"

namespace dragon {

template <class Context>
template <typename T>
void DropBlock2dOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    int64_t feature_h, feature_w, seed_h, seed_w;
    if (data_format() == "NCHW") {
      feature_h = X.dim(2), feature_w = X.dim(3);
    } else if (data_format() == "NHWC") {
      feature_h = X.dim(1), feature_w = X.dim(2);
    } else {
      LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }
    seed_h = feature_h - block_size_ + 1;
    seed_w = feature_w - block_size_ + 1;
    CHECK(seed_h > 0 && seed_w > 0) << "\nExcepted block_size <= feature_size.";
    // Compute the drop ratio
    float gamma = ratio() / std::pow(block_size_, 2);
    gamma *= (float(feature_h * feature_w) / float(seed_h * seed_w));
    // Prepare buffers
    auto* mask = Buffer("mask")
                     ->ReshapeLike(X)
                     ->template mutable_data<uint8_t, Context>();
    auto* scale = Buffer("scale")
                      ->Reshape({})
                      ->template mutable_data<float, CPUContext>();
    auto scratches = ctx()->workspace()->template data<Context>({
        X.dim(0) * seed_h * seed_w * sizeof(uint32_t), // seed points
        X.count() * sizeof(int), // int32 mask for seed growing
    });
    // Fill mask with ones
    math::Set(X.count(), 1, (int*)scratches[1], ctx());
    // Generate 2d mask from the seed region
    kernel::DropBlock2d(
        X.dim(0),
        data_format() == "NCHW" ? X.dim(1) : X.dim(-1),
        feature_h,
        feature_w,
        seed_h,
        seed_w,
        block_size_,
        gamma,
        data_format(),
        (uint32_t*)scratches[0],
        (int*)scratches[1],
        ctx());
    // Convert to uint8 mask
    math::Cast(X.count(), (int*)scratches[1], mask, ctx());
    // Count the number of zeros to compute scale factor
    float normalizer = math::Sum(X.count(), 1.f, (int*)scratches[1], ctx());
    scale[0] = (float)X.count() / std::max(normalizer, 1.f);
    // Apply mask to the feature
    kernel::ApplyMask(
        X.count(),
        scale[0],
        X.template data<T, Context>(),
        mask,
        Y->ReshapeLike(X)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void DropBlock2dOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void DropBlock2dGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  if (phase() == "TEST") {
    NOT_IMPLEMENTED;
  } else if (phase() == "TRAIN") {
    kernel::ApplyMask(
        dY.count(),
        Buffer("scale")->template data<float, CPUContext>()[0],
        dY.template data<T, Context>(),
        Buffer("mask")->template data<uint8_t, Context>(),
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unknown Phase: " << phase();
  }
}

template <class Context>
void DropBlock2dGradientOp<Context>::RunOnDevice() {
  DispatchHelper<FloatingTensorTypes>::Call(this, Input(0));
}

DEPLOY_CPU_OPERATOR(DropBlock2d);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DropBlock2d);
#endif

OPERATOR_SCHEMA(DropBlock2d)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

DEPLOY_CPU_OPERATOR(DropBlock2dGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DropBlock2dGradient);
#endif

OPERATOR_SCHEMA(DropBlock2dGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(DropBlock2d, SimpleGradientMaker);

} // namespace dragon
