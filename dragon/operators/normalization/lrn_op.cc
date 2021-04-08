#include "dragon/operators/normalization/lrn_op.h"

namespace dragon {

template <class Context>
template <typename T>
void LRNOp<Context>::DoRunWithType() {
  LOG(FATAL) << "Compile with CuDNN for LocalResponseNorm.";
}

template <class Context>
void LRNOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(0));
  DispatchHelper<dtypes::Floating>::Call(this, Input(0));
}

template <class Context>
template <typename T>
void LRNGradientOp<Context>::DoRunWithType() {
  LOG(FATAL) << "Compile with CuDNN for LocalResponseNorm.";
}
template <class Context>
void LRNGradientOp<Context>::RunOnDevice() {
  Output(0)->ReshapeLike(Input(-1));
  DispatchHelper<dtypes::Floating>::Call(this, Input(-1));
}

DEPLOY_CPU_OPERATOR(LRN);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LRN);
#endif

DEPLOY_CPU_OPERATOR(LRNGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LRNGradient);
#endif

OPERATOR_SCHEMA(LRN)
    /* X */
    .NumInputs(1)
    /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(LRNGradient)
    /* X, Y, dY */
    .NumInputs(3)
    /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);
  void CreateGradientDefs() override {
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({I(0), O(0), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

REGISTER_GRADIENT(LRN, GradientMaker);

} // namespace dragon
