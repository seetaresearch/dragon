#include "core/workspace.h"
#include "operators/misc/initialize_op.h"

namespace dragon {

template <class Context> template <typename T>
void InitializeOp<Context>::RunWithType() {
    unique_ptr< Filler<T, Context> > f;
    f.reset(CreateFiller<T, Context>(filler_proto));
    f->Fill(Output(0), ctx());
}

template <class Context>
void InitializeOp<Context>::RunOnDevice() {
    vector<int64_t> output_shape;
    if (shape_desc.empty()) {
        // Determine the shape from dimensions
        int ndims = (int)std::max(dims_value.size(), dims_desc.size());
        for (int i = 0; i < ndims; i++) output_shape.push_back(dims(i));
    } else {
        // Determine the shape from given shape
        Tensor* shape = ws()->GetTensor(shape_desc);
        CHECK(shape->IsType<int64_t>()) << "\nThe type of shape should be int64.";
        auto* shape_data = shape->template data<int64_t, CPUContext>();
        for (int i = 0; i < shape->count(); i++) output_shape.push_back(shape_data[i]);
    }
    Output(0)->Reshape(output_shape);
    RunWithType<float>();
}

template <class Context> template <typename T>
void FillOp<Context>::RunWithType() {
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    math::Set(Output(0)->count(), cast::to<T>(value), Ydata, ctx());
}

template <class Context>
void FillOp<Context>::RunOnDevice() {
    vector<int64_t> output_shape;
    if (shape_desc.empty()) {
        // Determine the shape from the given dimensions
        int ndims = (int)std::max(dims_value.size(), dims_desc.size());
        for (int i = 0; i < ndims; i++) output_shape.push_back(dims(i));
    } else {
        // Determine the shape from the given shape
        Tensor* shape = ws()->GetTensor(shape_desc);
        CHECK(shape->IsType<int64_t>()) << "\nThe type of shape should be int64.";
        auto* shape_data = shape->template data<int64_t, CPUContext>();
        for (int i = 0; i < shape->count(); i++) output_shape.push_back(shape_data[i]);
    }
    Output(0)->Reshape(output_shape);
    if (dtype == "bool") RunWithType<bool>();
    else if (dtype == "int8") RunWithType<int8_t>();
    else if (dtype == "uint8") RunWithType<uint8_t>();
    else if (dtype == "int32") RunWithType<int>();
    else if (dtype == "int64") RunWithType<int64_t>();
    else if (dtype == "float16") RunWithType<float16>();
    else if (dtype == "float32") RunWithType<float>();
    else if (dtype == "float64") RunWithType<double>();
    else LOG(FATAL) << DTypeHelper(dtype, {
        "bool", "int8", "uint8", "int32", "int64",
            "float16", "float32", "float64",
    });
}

template <class Context> template <typename T>
void GivenTensorFillOp<Context>::RunWithType() {
    ExtractValues<T>();

    CHECK_EQ(Output(0)->count(), values.count())
        << "\nExcepted the size of given values is "
        << Output(0)->count() << ", but got " << values.count();

    auto* Xdata = values.template data<T, CPUContext>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    ctx()->template Copy<T, Context, CPUContext>(
        values.count(), Ydata, Xdata);
}

template <class Context>
void GivenTensorFillOp<Context>::RunOnDevice() {
    vector<int64_t> output_shape;
    if (shape.empty()) {
        // Determine the shape from the given dimensions
        int ndims = (int)std::max(dims_value.size(), dims_desc.size());
        for (int i = 0; i < ndims; i++) output_shape.push_back(dims(i));
        // Regard it as a Vector if missing the dimensions
        if (ndims == 0) output_shape.push_back(values.count());
    } else {
        // Determine the shape from the given shape
        output_shape = shape;
    }
    Output(0)->Reshape(output_shape);

    if (dtype == "int32") RunWithType<int>();
    else if (dtype == "int64") RunWithType<int64_t>();
    else if (dtype == "float16") RunWithType<float16>();
    else if (dtype == "float32") RunWithType<float>();
    else if (dtype == "float64") RunWithType<double>();
    else if (dtype == "string") RunWithType<string>();
    else LOG(FATAL) << DTypeHelper(dtype, {
        "int32", "int64", "string",
            "float16", "float32", "float64",
    });
}

// ConstantFill
DEPLOY_CPU(Fill);
#ifdef WITH_CUDA
DEPLOY_CUDA(Fill);
#endif
OPERATOR_SCHEMA(Fill).NumInputs(0).NumOutputs(1);
NO_GRADIENT(Fill);

// GivenTensorFill
DEPLOY_CPU(GivenTensorFill);
#ifdef WITH_CUDA
DEPLOY_CUDA(GivenTensorFill);
#endif
OPERATOR_SCHEMA(GivenTensorFill).NumInputs(0).NumOutputs(1);
NO_GRADIENT(GivenTensorFill);

// Uniform
DEPLOY_CPU(RandomUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomUniform);
#endif
OPERATOR_SCHEMA(RandomUniform).NumInputs(0).NumOutputs(1);
NO_GRADIENT(RandomUniform);

// Normal
DEPLOY_CPU(RandomNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomNormal);
#endif
OPERATOR_SCHEMA(RandomNormal).NumInputs(0).NumOutputs(1);
NO_GRADIENT(RandomNormal);

// TruncatedNormal
#ifdef WITH_CUDA
DEPLOY_CPU_CUDA(TruncatedNormal);
#else
DEPLOY_CPU(TruncatedNormal);
#endif
OPERATOR_SCHEMA(TruncatedNormal).NumInputs(0).NumOutputs(1);
NO_GRADIENT(TruncatedNormal);

// GlorotUniform
DEPLOY_CPU(GlorotUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotUniform);
#endif
OPERATOR_SCHEMA(GlorotUniform).NumInputs(0).NumOutputs(1);
NO_GRADIENT(GlorotUniform);

// GlorotNormal
DEPLOY_CPU(GlorotNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotNormal);
#endif
OPERATOR_SCHEMA(GlorotNormal).NumInputs(0).NumOutputs(1);
NO_GRADIENT(GlorotNormal);

}  // namespace dragon