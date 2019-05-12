#include "core/workspace.h"
#include "operators/misc/initialize_op.h"

namespace dragon {

template <class Context> template <typename T>
void InitializeOp<Context>::RunImpl() {
    unique_ptr< Filler<T, Context> > f;
    f.reset(CreateFiller<T, Context>(proto_));
    f->Fill(Y(0), ctx());
}

template <class Context>
void InitializeOp<Context>::RunOnDevice() {
    vec64_t out_shape;
    if (shape_desc_.empty()) {
        // Determine the shape from dimensions
        int ndims = GET_ARGS_SIZE(dims);
        for (int i = 0; i < ndims; i++)
            out_shape.push_back(dims(i));
    } else {
        // Determine the shape from given shape
        auto* t = ws()->GetTensor(shape_desc_);
        auto* shape = t->template data<int64_t, CPUContext>();
        for (int i = 0; i < t->count(); i++)
            out_shape.push_back(shape[i]);
    }
    Y(0)->Reshape(out_shape);
    RunImpl<float>();
}

template <class Context> template <typename T>
void FillOp<Context>::RunImpl() {
    auto* y = Y(0)->template
        mutable_data<T, Context>();

    math::Set(
        Y(0)->count(),
        cast::to<T>(value_),
        y, ctx()
    );
}

template <class Context>
void FillOp<Context>::RunOnDevice() {
    vec64_t out_shape;
    if (shape_desc_.empty()) {
        // Determine the shape from dimensions
        int ndims = GET_ARGS_SIZE(dims);
        for (int i = 0; i < ndims; i++)
            out_shape.push_back(dims(i));
    } else {
        // Determine the shape from given shape
        auto* sl = ws()->GetTensor(shape_desc_);
        auto* shape = sl->template data<int64_t, CPUContext>();
        for (int i = 0; i < sl->count(); i++)
            out_shape.push_back(shape[i]);
    }

    Y(0)->Reshape(out_shape);

    if (dtype() == "bool") {
        RunImpl<bool>();
    } else if (dtype() == "int8") {
        RunImpl<int8_t>();
    } else if (dtype() == "uint8") {
        RunImpl<uint8_t>();
    } else if (dtype() == "int32") {
        RunImpl<int>();
    } else if (dtype() == "int64") {
        RunImpl<int64_t>();
    } else if (dtype() == "float16"){
        RunImpl<float16>();
    } else if (dtype() == "float32") {
        RunImpl<float>();
    } else if (dtype() == "float64") {
        RunImpl<double>();
    } else {
        LOG(FATAL) << DTypeString(dtype(), {
            "bool", "int8", "uint8", "int32", "int64",
                "float16", "float32", "float64",
        });
    }
}

template <class Context> template <typename T>
void GivenTensorFillOp<Context>::RunImpl() {
    Extract<T>();

    CHECK_EQ(Y(0)->count(), values_.count())
        << "\nExcepted the size of output is "
        << values_.count() << ", while got "
        << Y(0)->count();

    auto* x = values_.template data<T, CPUContext>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    ctx()->template Copy
        <T, Context, CPUContext>(
            values_.count(), y, x);
}

template <class Context>
void GivenTensorFillOp<Context>::RunOnDevice() {
    vec64_t out_shape;
    if (shape_.empty()) {
        // Determine the shape from the given dimensions
        int ndims = GET_ARGS_SIZE(dims);
        for (int i = 0; i < ndims; i++)
            out_shape.push_back(dims(i));
        // Regard it as a Vector if missing the dimensions
        if (ndims == 0) out_shape.push_back(values_.count());
    } else {
        // Determine the shape from the given shape
        out_shape = shape_;
    }

    Y(0)->Reshape(out_shape);

    if (dtype() == "int32") {
        RunImpl<int>();
    } else if (dtype() == "int64") {
        RunImpl<int64_t>();
    } else if (dtype() == "float16") {
        RunImpl<float16>();
    } else if (dtype() == "float32") {
        RunImpl<float>();
    } else if (dtype() == "float64") {
        RunImpl<double>();
    } else if (dtype() == "string") {
        RunImpl<string>();
    } else {
        LOG(FATAL) << DTypeString(dtype(), {
            "int32", "int64", "string",
            "float16", "float32", "float64",
        });
    }
}

DEPLOY_CPU(Fill);
#ifdef WITH_CUDA
DEPLOY_CUDA(Fill);
#endif

DEPLOY_CPU(GivenTensorFill);
#ifdef WITH_CUDA
DEPLOY_CUDA(GivenTensorFill);
#endif

DEPLOY_CPU(RandomUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomUniform);
#endif

DEPLOY_CPU(RandomNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(RandomNormal);
#endif

#ifdef WITH_CUDA
DEPLOY_CPU_CUDA(TruncatedNormal);
#else
DEPLOY_CPU(TruncatedNormal);
#endif

DEPLOY_CPU(GlorotUniform);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotUniform);
#endif

DEPLOY_CPU(GlorotNormal);
#ifdef WITH_CUDA
DEPLOY_CUDA(GlorotNormal);
#endif

OPERATOR_SCHEMA(Fill)
    .NumInputs(0)
    .NumOutputs(1);

OPERATOR_SCHEMA(GivenTensorFill)
    .NumInputs(0)
    .NumOutputs(1);

OPERATOR_SCHEMA(RandomUniform)
    .NumInputs(0)
    .NumOutputs(1);

OPERATOR_SCHEMA(RandomNormal)
    .NumInputs(0)
    .NumOutputs(1);

OPERATOR_SCHEMA(TruncatedNormal)
    .NumInputs(0)
    .NumOutputs(1);

OPERATOR_SCHEMA(GlorotUniform)
    .NumInputs(0)
    .NumOutputs(1);

OPERATOR_SCHEMA(GlorotNormal)
    .NumInputs(0).NumOutputs(1);

NO_GRADIENT(Fill);
NO_GRADIENT(GivenTensorFill);
NO_GRADIENT(RandomUniform);
NO_GRADIENT(RandomNormal);
NO_GRADIENT(TruncatedNormal);
NO_GRADIENT(GlorotUniform);
NO_GRADIENT(GlorotNormal);

}  // namespace dragon