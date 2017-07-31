#include "operators/vision/dense_concat_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

template <class Context>
void DenseConcatOp<Context>::RunOnDevice() {
    ConcatOp<Context>::RunOnDevice();
    input(0).Release();  // keep shape, just release mem
}

DEPLOY_CPU(DenseConcat);
#ifdef WITH_CUDA
DEPLOY_CUDA(DenseConcat);
#endif
OPERATOR_SCHEMA(DenseConcat).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void DenseConcatGradientOp<Context>::RunWithType() {
    //  restore X1 from Y
    auto* Ydata = input(-2).template data<T, Context>();
    auto* Xdata = input(0).template mutable_data<T, Context>();
    this->x_concat_dim = input(0).dim(this->axis);
    TIndex count = input(0).count();
    this->concat_dims = input(-1).dims();
    this->y_concat_dim = this->concat_dims[this->axis];
    this->outer_dim = input(-1).count(0, this->axis);
    this->inner_dim = input(-1).count(this->axis + 1);
    kernel::ConcatGrad<T, Context>(count,
                         this->outer_dim, 
                         this->inner_dim,
                      this->x_concat_dim, 
                      this->y_concat_dim,
                                       0,
                                   Ydata, 
                                   Xdata,
                                 &ctx());
}

template <class Context>
void DenseConcatGradientOp<Context>::RunOnDevice() {
    if (input(0).template IsType<float>()) RunWithType<float>();
    else if (input(0).template IsType<float16>()) RunWithType<float16>();
    else LOG(FATAL) << "unsupported input types.";

    ConcatGradientOp<Context>::RunOnDevice();
}

template <class Context>
void DenseConcatGradientOp<Context>::ShareBeforeRun() {
    Tensor* dX = ws()->GetBuffer();
    if (dX != nullptr) output(0)->Replace(*dX);
}

template <class Context>
void DenseConcatGradientOp<Context>::ClearAfterRun() {
    Tensor* dY = &input(-1);
    Tensor* Y = &input(-2);
    ws()->ReleaseBuffer(dY);
    ws()->ReleaseBuffer(Y, true);
}

DEPLOY_CPU(DenseConcatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DenseConcatGradient);
#endif
OPERATOR_SCHEMA(DenseConcatGradient).NumInputs(4).NumOutputs(2);

class GetDenseConcatGradient : public GradientMakerBase {
public:
    GRADIENT_MAKER_CTOR(GetDenseConcatGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), O(0), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(DenseConcat, GetDenseConcatGradient);

}   // namespace dragon