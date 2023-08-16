#include "dragon/operators/activation/dropblock_op.h"
#include "dragon/core/workspace.h"
#include "dragon/kernels/op_kernels.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <class Context>
template <typename T>
void DropBlockOp<Context>::DoRunWithType() {
  auto &X = Input(0), *Y = Output(0, {0});
  if (phase() == "TEST") {
    Y->ReshapeLike(X)->CopyFrom(X, ctx());
  } else if (phase() == "TRAIN") {
    const auto axis = data_format() == "NCHW" ? 2 : 1;
    const auto num_axes = X.ndim() - 2;
    int64_t feat_area = 1, mask_area = 1, N = X.dim(0);
    for (int i = axis; i < axis + num_axes; ++i) {
      auto feat_size = X.dim(i);
      auto seed_size = feat_size - block_size_ + 1;
      CHECK(seed_size > 0) << "\nExcepted block_size <= feat_size";
      feat_area *= feat_size;
      mask_area *= block_size_ * seed_size;
      N *= seed_size;
    }
    const auto alpha = ratio();
    const auto gamma = alpha * float(feat_area) / float(mask_area);
    auto* X_mask = Output("X_mask")->ReshapeLike(X);
    auto* scratch = ctx()->workspace()->template data<float, Context>(N);
    math::RandomUniform(N, 0.f, 1.f, scratch, ctx());
    if (num_axes == 1 || num_axes == 2) {
      kernels::DropBlock2d(
          X.dim(0),
          data_format() == "NCHW" ? X.dim(1) : X.dim(-1),
          X.dim(axis),
          num_axes == 1 ? 1 : X.dim(axis + 1),
          block_size_,
          gamma, // ratio
          1.f / (1.f - alpha), // scale
          data_format(),
          scratch,
          X.template data<T, Context>(),
          Y->ReshapeLike(X)->template mutable_data<T, Context>(),
          X_mask->template mutable_data<uint8_t, Context>(),
          ctx());
    }
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

template <class Context>
template <typename T>
void DropBlockGradientOp<Context>::DoRunWithType() {
  auto &dY = Input(0), *dX = Output(0);
  if (phase() == "TRAIN") {
    math::ApplyMask(
        dY.count(),
        1.f / (1.f - ratio()),
        Input("X_mask").template data<uint8_t, Context>(),
        dY.template data<T, Context>(),
        dX->ReshapeLike(dY)->template mutable_data<T, Context>(),
        ctx());
  } else {
    LOG(FATAL) << "Unsupported phase: " << phase();
  }
}

DEPLOY_CPU_OPERATOR(DropBlock);
DEPLOY_CPU_OPERATOR(DropBlockGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(DropBlock);
DEPLOY_CUDA_OPERATOR(DropBlockGradient);
#endif

DEFINE_OP_SINGLE_ARG(float, DropBlockOp, ratio);
DEFINE_OP_SINGLE_ARG(float, DropBlockGradientOp, ratio);

OPERATOR_SCHEMA(DropBlock)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1)
    /* X => Y */
    .AllowInplace({{0, 0}});

OPERATOR_SCHEMA(DropBlockGradient)
    /* dY */
    .NumInputs(1)
    /* dX */
    .NumOutputs(1)
    /* dY => dX */
    .AllowInplace({{0, 0}});

REGISTER_GRADIENT(DropBlock, SimpleGradientMaker);

} // namespace dragon
