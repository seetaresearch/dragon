#include "operators/utils/initialize_op.h"
#include "core/workspace.h"

namespace dragon {

template <class Context> template <typename T>
void InitializeOp<Context>::RunWithType() {
    unique_ptr< Filler<T, Context> > f;
    f.reset(CreateFiller<T, Context>(filler));
    f->Fill(output(0));
}

template <class Context>
void InitializeOp<Context>::RunOnDevice() {
    vector<TIndex> dims;
    if (dynamic_shape.empty()) {
        for (auto& dim : static_shape) dims.push_back(dim);
    } else {
        auto* shape_data = ws()->GetTensor(dynamic_shape)
                               ->template data<float, CPUContext>();
        TIndex ndim = ws()->GetTensor(dynamic_shape)->count();
        for (int i = 0; i < ndim; i++) dims.push_back(shape_data[i]);
    }
    output(0)->Reshape(dims);

    RunWithType<float>();
}

//    constant
DEPLOY_CPU(Fill);
#ifdef WITH_CUDA
DEPLOY_CUDA(Fill);
#endif
OPERATOR_SCHEMA(Fill).NumInputs(0, 1).NumOutputs(1);
NO_GRADIENT(Fill);

//    uniform
DEPLOY_CPU(RandomUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomUniform);
#endif
OPERATOR_SCHEMA(RandomUniform).NumInputs(0, 1).NumOutputs(1);
NO_GRADIENT(RandomUniform);

//    normal
DEPLOY_CPU(RandomNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomNormal);
#endif
OPERATOR_SCHEMA(RandomNormal).NumInputs(0, 1).NumOutputs(1);
NO_GRADIENT(RandomNormal);

//    truncated normal
DEPLOY_CPU(TruncatedNormal);
#ifdef WITH_CUDA
DEPLOY_CPU_CUDA(TruncatedNormal);
#endif
OPERATOR_SCHEMA(TruncatedNormal).NumInputs(0, 1).NumOutputs(1);
NO_GRADIENT(TruncatedNormal);

//    glorot uniform
DEPLOY_CPU(GlorotUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotUniform);
#endif
OPERATOR_SCHEMA(GlorotUniform).NumInputs(0, 1).NumOutputs(1);
NO_GRADIENT(GlorotUniform);

//    glorot normal
DEPLOY_CPU(GlorotNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotNormal);
#endif
OPERATOR_SCHEMA(GlorotNormal).NumInputs(0, 1).NumOutputs(1);
NO_GRADIENT(GlorotNormal);

}    // namespace dragon