#include "dragon/operators/normalization/lrn_op.h"

namespace dragon {

template <class Context>
template <typename T>
void LRNOp<Context>::DoRunWithType() {
  LOG(FATAL) << "Compile with CuDNN for LocalResponseNorm.";
}

template <class Context>
template <typename T>
void LRNGradientOp<Context>::DoRunWithType() {
  LOG(FATAL) << "Compile with CuDNN for LocalResponseNorm.";
}

DEPLOY_CPU_OPERATOR(LRN);
DEPLOY_CPU_OPERATOR(LRNGradient);
#ifdef USE_CUDA
DEPLOY_CUDA_OPERATOR(LRN);
DEPLOY_CUDA_OPERATOR(LRNGradient);
#endif

OPERATOR_SCHEMA(LRN).NumInputs(1).NumOutputs(1);
OPERATOR_SCHEMA(LRNGradient).NumInputs(3).NumOutputs(1);

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
