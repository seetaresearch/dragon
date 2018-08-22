#include "core/workspace.h"
#include "operators/misc/initialize_op.h"

namespace dragon {

template <class Context> template <typename T>
void InitializeOp<Context>::RunWithType() {
    unique_ptr< Filler<T, Context> > f;
    f.reset(CreateFiller<T, Context>(filler));
    f->Fill(Output(0), ctx());
}

template <class Context>
void InitializeOp<Context>::RunOnDevice() {
    vector<TIndex> output_shape;
    if (shape_desc.empty()) {
        //  determine the shape from dimensions
        int ndims = (int)std::max(dims_value.size(), dims_desc.size());
        for (int i = 0; i < ndims; i++) output_shape.push_back(dims(i));
    } else {
        //  determine the shape from given shape
        Tensor* shape = ws()->GetTensor(shape_desc);
        CHECK(shape->IsType<int>()) << "\nThe type of shape should be int32.";
        auto* shape_data = shape->template data<int, CPUContext>();
        for (int i = 0; i < shape->count(); i++) output_shape.push_back(shape_data[i]);
    }
    Output(0)->Reshape(output_shape);
    RunWithType<float>();
}

//  constant
DEPLOY_CPU(Fill);
#ifdef WITH_CUDA
DEPLOY_CUDA(Fill);
#endif
OPERATOR_SCHEMA(Fill).NumInputs(0).NumOutputs(1);
NO_GRADIENT(Fill);

//  uniform
DEPLOY_CPU(RandomUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomUniform);
#endif
OPERATOR_SCHEMA(RandomUniform).NumInputs(0).NumOutputs(1);
NO_GRADIENT(RandomUniform);

//  normal
DEPLOY_CPU(RandomNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomNormal);
#endif
OPERATOR_SCHEMA(RandomNormal).NumInputs(0).NumOutputs(1);
NO_GRADIENT(RandomNormal);

//  truncated normal
#ifdef WITH_CUDA
DEPLOY_CPU_CUDA(TruncatedNormal);
#else
DEPLOY_CPU(TruncatedNormal);
#endif
OPERATOR_SCHEMA(TruncatedNormal).NumInputs(0).NumOutputs(1);
NO_GRADIENT(TruncatedNormal);

//  glorot uniform
DEPLOY_CPU(GlorotUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotUniform);
#endif
OPERATOR_SCHEMA(GlorotUniform).NumInputs(0, 1).NumOutputs(1);
NO_GRADIENT(GlorotUniform);

//  glorot normal
DEPLOY_CPU(GlorotNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotNormal);
#endif
OPERATOR_SCHEMA(GlorotNormal).NumInputs(0).NumOutputs(1);
NO_GRADIENT(GlorotNormal);

}    // namespace dragon