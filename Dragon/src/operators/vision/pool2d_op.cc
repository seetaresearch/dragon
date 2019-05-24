#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/pool_op.h"

namespace dragon {

#define DEFINE_SAME_PADDING(A, B) \
    A[i] = padding_needed / 2; \
    B[i] = padding_needed - A[i]

#define DETERMINE_RUNTIME_ARGS \
    if (data_format() == "NCHW") { \
        n_ = X(0).dim(0), c_ = X(0).dim(1); \
        h_ = X(0).dim(2), w_ = X(0).dim(3); \
        if (global_pool_) { \
            for (int i = 0; i < 2; i++) \
                kshape_[i] = X(0).dim(i + 2); \
        } \
        if (str::find(padding_, "SAME")) { \
            for (int i = 0; i < 2; i++) { \
                auto idm = X(0).dim(i + 2); \
                int64_t odm = (idm + stride_[i] - 1 \
                    ) / (float)stride_[i]; \
                auto padding_needed = std::max((int64_t)0, \
                    (odm - 1) * stride_[i] + kshape_[i] - idm); \
                if (padding_ != "SAME_UPPER") { \
                    DEFINE_SAME_PADDING(pad_l_, pad_r_); \
                } else { \
                    DEFINE_SAME_PADDING(pad_r_, pad_l_); \
                }  /*! SAME_LOWER or SAME */ \
            } \
        } \
    } else if (data_format() == "NHWC") { \
        n_ = X(0).dim(0), h_ = X(0).dim(1); \
        w_ = X(0).dim(2), c_ = X(0).dim(3); \
        if (global_pool_) { \
            for (int i = 0; i < 2; i++) \
                kshape_[i] = X(0).dim(i + 1); \
        } \
        if (str::find(padding_, "SAME")) { \
            for (int i = 0; i < 2; i++) { \
                auto idm = X(0).dim(i + 1); \
                int64_t odm = (idm + stride_[i] - 1 \
                    ) / (float)stride_[i]; \
                auto padding_needed = std::max((int64_t)0, \
                   (odm - 1) * stride_[i] + kshape_[i] - idm); \
                if (padding_ != "SAME_UPPER") { \
                    DEFINE_SAME_PADDING(pad_l_, pad_r_); \
                } else { \
                    DEFINE_SAME_PADDING(pad_r_, pad_l_); \
                }  /*! SAME_LOWER or SAME */ \
            } \
        } \
    } else { \
        LOG(FATAL) << "Unknown DataFormat: " << data_format(); \
    } \
    if (!str::find(padding_, "SAME")) { \
        /*! Case 1: infer output shape with explicit pads */ \
        if (ceil_mode_) { \
            pool_h_ = ceil( \
                (h_ + pad_l_[0] + pad_r_[0] - kshape_[0] \
                    ) / (float)stride_[0]) + 1; \
            pool_w_ = ceil( \
                (w_ + pad_l_[1] + pad_r_[1] - kshape_[1] \
                    ) / (float)stride_[1]) + 1; \
        } else { \
            pool_h_ = floor( \
                (h_ + pad_l_[0] + pad_r_[0] - kshape_[0] \
                    ) / (float)stride_[0]) + 1; \
            pool_w_ = floor( \
                (w_ + pad_l_[1] + pad_r_[1] - kshape_[1] \
                    ) / (float)stride_[1]) + 1; \
        } \
        if ((pool_h_ - 1) * stride_[0] >= \
                (h_ + pad_l_[0] + pad_r_[0])) pool_h_--; \
        if ((pool_w_ - 1) * stride_[1] >= \
                (w_ + pad_l_[1] + pad_r_[1])) pool_w_--; \
    } else { \
        /*! Case 2: infer output shape with auto pads */ \
        pool_h_ = ceil((float)h_ / (float)stride_[0]); \
        pool_w_ = ceil((float)w_ / (float)stride_[1]); \
    }

template <class Context> template <typename T>
void Pool2dOp<Context>::MaxRunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    auto* mask = ws()->CreateTensor(
        unique_name("mask")
            )->ReshapeLike(*Y(0)
                )->template mutable_data<int, Context>();

    kernel::MaxPool2d(
        n_, c_, h_, w_,
        pool_h_, pool_w_,
        kshape_[0], kshape_[1],
        stride_[0], stride_[1],
        pad_l_[0], pad_l_[1],
        data_format(),
        x, mask, y, ctx()
    );
}

template <class Context> template <typename T>
void Pool2dOp<Context>::AvgRunImpl() {
    auto* x = X(0).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::AvgPool2d(
        n_, c_, h_, w_,
        pool_h_, pool_w_,
        kshape_[0], kshape_[1],
        stride_[0], stride_[1],
        pad_l_[0], pad_l_[1],
        data_format(),
        x, y, ctx()
    );
}

template <class Context>
void Pool2dOp<Context>::Reshape() {
    DETERMINE_RUNTIME_ARGS;
    if (data_format() == "NCHW") {
        Y(0)->Reshape(
            vec64_t({ n_, c_, pool_h_, pool_w_ })
        );
    } else if (data_format() == "NHWC") {
        Y(0)->Reshape(
            vec64_t({ n_, pool_h_, pool_w_, c_ })
        );
    }
}

template <class Context>
void Pool2dOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        if (mode_ == "MAX") {
            MaxRunImpl<float>();
        } else if (mode_ == "AVG") {
            AvgRunImpl<float>();
        }
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

template <class Context> template <typename T>
void Pool2dGradientOp<Context>::MaxRunImpl() {
    auto* mask = ws()->GetTensor(
        unique_name("mask")
            )->template data<int, Context>();

    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::MaxPool2dGrad(
        n_, c_, h_, w_, pool_h_, pool_w_,
        kshape_[0], kshape_[1],
        stride_[0], stride_[1],
        pad_l_[0], pad_l_[1],
        data_format(),
        dy, mask, dx, ctx()
    );
}

template <class Context> template <typename T>
void Pool2dGradientOp<Context>::AvgRunImpl() {
    auto* dy = X(-1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    kernel::AvgPool2dGrad(
        n_, c_, h_, w_, pool_h_, pool_w_,
        kshape_[0], kshape_[1],
        stride_[0], stride_[1],
        pad_l_[0], pad_l_[1],
        data_format(),
        dy, dx, ctx()
    );
}

template <class Context>
void Pool2dGradientOp<Context>::Reshape() {
    DETERMINE_RUNTIME_ARGS;
    Y(0)->ReshapeLike(X(0));
}

template <class Context>
void Pool2dGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        if (mode_ == "MAX") {
            MaxRunImpl<float>();
        } else if (mode_ == "AVG") {
            AvgRunImpl<float>();
        }
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(Pool2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pool2d);
#endif

DEPLOY_CPU(Pool2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(Pool2dGradient);
#endif

OPERATOR_SCHEMA(Pool2d)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(Pool2dGradient)
     /* X, Y, dY */
    .NumInputs(3)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), O(0), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

}  // namespace

REGISTER_GRADIENT(Pool2d, GradientMaker);

#undef DETERMINE_RUNTIME_ARGS

}  // namespace dragon