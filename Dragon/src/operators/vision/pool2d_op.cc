#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/pool_op.h"

namespace dragon {

#define DEFINE_SAME_PADDING(A, B) \
    A[i] = padding_needed / 2; \
    B[i] = padding_needed - A[i]

#define DETERMINE_RUNTIME_ARGUMENTS(X) \
    if (data_format == "NCHW") { \
        n = Input(0).dim(0), c = Input(0).dim(1); \
        h = Input(0).dim(2), w = Input(0).dim(3); \
        if (global_pooling) { \
            for (int i = 0; i < 2; i++) \
                kernel_shape[i] = Input(0).dim(i + 2); \
        } \
        if (padding.find("SAME") != string::npos) { \
            for (int i = 0; i < 2; i++) { \
                int64_t idm = Input(0).dim(i + 2); \
                int64_t odm = (idm + stride[i] - 1) / (float)stride[i]; \
                int64_t padding_needed = std::max((int64_t)0, \
                    (odm - 1) * stride[i] + kernel_shape[i] - idm); \
                if (padding != "SAME_UPPER") { DEFINE_SAME_PADDING(pad_l, pad_r); } \
                else { DEFINE_SAME_PADDING(pad_r, pad_l); } /*! SAME_LOWER or SAME */ \
            } \
        } \
    } else if (data_format == "NHWC") { \
        n = Input(0).dim(0), h = Input(0).dim(1); \
        w = Input(0).dim(2), c = Input(0).dim(3); \
        if (global_pooling) { \
            for (int i = 0; i < 2; i++) \
                kernel_shape[i] = Input(0).dim(i + 1); \
        } \
        if (padding.find("SAME") != string::npos) { \
            for (int i = 0; i < 2; i++) { \
                int64_t idm = Input(0).dim(i + 1); \
                int64_t odm = (idm + stride[i] - 1) / (float)stride[i]; \
                int64_t padding_needed = std::max((int64_t)0, \
                   (odm - 1) * stride[i] + kernel_shape[i] - idm); \
                if (padding != "SAME_UPPER") { DEFINE_SAME_PADDING(pad_l, pad_r); } \
                else { DEFINE_SAME_PADDING(pad_r, pad_l); }  /*! SAME_LOWER or SAME */ \
            } \
        } \
    } else LOG(FATAL) << "Unknown data format: " << data_format; \
    if (padding.find("SAME") == string::npos) { \
        /*! Case 1: infer output shape with explicit pads */ \
        if (ceil_mode) { \
            pool_h = ceil((h + pad_l[0] + pad_r[0] - kernel_shape[0]) / (float)stride[0]) + 1; \
            pool_w = ceil((w + pad_l[1] + pad_r[1] - kernel_shape[1]) / (float)stride[1]) + 1; \
        } else { \
            pool_h = floor((h + pad_l[0] + pad_r[0] - kernel_shape[0]) / (float)stride[0]) + 1; \
            pool_w = floor((w + pad_l[1] + pad_r[1] - kernel_shape[1]) / (float)stride[1]) + 1; \
        } \
        if ((pool_h - 1) * stride[0] >= (h + pad_l[0] + pad_r[0])) pool_h--; \
        if ((pool_w - 1) * stride[1] >= (w + pad_l[1] + pad_r[1])) pool_w--; \
    } else { \
        /*! Case 2: infer output shape with auto pads */ \
        pool_h = ceil((float)h / (float)stride[0]); \
        pool_w = ceil((float)w / (float)stride[1]); \
    }

template <class Context> template <typename T>
void Pool2dOp<Context>::MAXRunWithType() {
    mask = ws()->CreateTensor(mount_name(
        "max_pool/mask"))->ReshapeLike(*Output(0));

    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template mutable_data<int, Context>();

    kernel::MAXPool2d(n, c, h, w, pool_h, pool_w,
        kernel_shape[0], kernel_shape[1],
            stride[0], stride[1], pad_l[0], pad_l[1],
                data_format, Xdata, Mdata, Ydata, ctx());
}

template <class Context> template <typename T>
void Pool2dOp<Context>::AVGRunWithType() {
    auto* Xdata = Input(0).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    kernel::AVGPool2d(n, c, h, w, pool_h, pool_w,
        kernel_shape[0], kernel_shape[1],
            stride[0], stride[1], pad_l[0], pad_l[1],
                data_format, Xdata, Ydata, ctx());
}

template <class Context>
void Pool2dOp<Context>::Reshape() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));
    if (data_format == "NCHW") {
        Output(0)->Reshape(vector<int64_t>({ n, c, pool_h, pool_w }));
    } else if (data_format == "NHWC") {
        Output(0)->Reshape(vector<int64_t>({ n, pool_h, pool_w, c }));
    }
}

template <class Context>
void Pool2dOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) {
        if (mode == "MAX") MAXRunWithType<float>();
        else if (mode == "AVG") AVGRunWithType<float>();
        else LOG(FATAL) << "Unsupported pooling mode: " << mode;
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Pool2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pool2d);
#endif
OPERATOR_SCHEMA(Pool2d).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void Pool2dGradientOp<Context>::MAXRunWithType() {
    mask = ws()->GetTensor(mount_name("max_pool/mask"));

    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* Mdata = mask->template data<int, Context>();

    kernel::MAXPool2dGrad(n, c, h, w, pool_h, pool_w,
        kernel_shape[0], kernel_shape[1],
            stride[0], stride[1], pad_l[0], pad_l[1],
                data_format, dYdata, Mdata, dXdata, ctx());
}

template <class Context> template <typename T>
void Pool2dGradientOp<Context>::AVGRunWithType() {
    auto* dYdata = Input(-1).template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();

    kernel::AVGPool2dGrad(n, c, h, w, pool_h, pool_w,
        kernel_shape[0], kernel_shape[1],
            stride[0], stride[1], pad_l[0], pad_l[1],
                data_format, dYdata, dXdata, ctx());
}

template <class Context>
void Pool2dGradientOp<Context>::Reshape() {
    DETERMINE_RUNTIME_ARGUMENTS(Input(0));
    Output(0)->ReshapeLike(Input(0));
}

template <class Context>
void Pool2dGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(Input(0), float)) {
        if (mode == "MAX") MAXRunWithType<float>();
        else if (mode == "AVG") AVGRunWithType<float>();
        else LOG(FATAL) << "Unsupported pooling mode: " << mode;
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(Pool2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pool2dGradient);
#endif

OPERATOR_SCHEMA(Pool2dGradient)
    .NumInputs(3).NumOutputs(1);

class GetPool2dGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetPool2dGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), O(0), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

REGISTER_GRADIENT(Pool2d, GetPool2dGradient);

#undef DETERMINE_RUNTIME_ARGUMENTS

}  // namespace dragon