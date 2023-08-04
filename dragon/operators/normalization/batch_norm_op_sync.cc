#include "dragon/operators/normalization/batch_norm_op.h"

namespace dragon {

REGISTER_CPU_OPERATOR(SyncBatchNorm, BatchNormOp<CPUContext>);
REGISTER_CPU_OPERATOR(SyncBatchNormGradient, BatchNormGradientOp<CPUContext>);
#ifdef USE_CUDA
REGISTER_CUDA_OPERATOR(SyncBatchNorm, BatchNormOp<CUDAContext>);
REGISTER_CUDA_OPERATOR(SyncBatchNormGradient, BatchNormGradientOp<CUDAContext>);
#endif

OPERATOR_SCHEMA(SyncBatchNorm)
    /* X, W, B, M, V */
    .NumInputs(5)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(SyncBatchNormGradient)
    /* X, W, M, V, dY */
    .NumInputs(5)
    /* dX, dW, dB */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), I(1), I(3), I(4), GO(0)}),
        vector<string>({GI(0), GI(1), GI(2)}));
  }
};

} // namespace

REGISTER_GRADIENT(SyncBatchNorm, GradientMaker);

} // namespace dragon
