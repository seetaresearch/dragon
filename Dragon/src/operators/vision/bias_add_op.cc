#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/bias_add_op.h"
#include "operators/arithmetic/affine_op.h"

namespace dragon {

template <class Context> template <typename T>
void BiasAddOp<Context>::RunImpl() {
    DECLARE_MULTIPLIER(multiplier, inner_dim_);
    TENSOR_FILL(X(1), vec64_t({ axis_dim_ }));

    // Copy X to Y firstly if necessary
    Y(0)->CopyFrom(X(0), ctx());

    auto* b = X(1).template data<T, Context>();
    auto* y = Y(0)->template mutable_data<T, Context>();

    kernel::BiasAdd(
        outer_dim_,
        axis_dim_,
        inner_dim_,
        data_format(),
        b, multiplier,
        y, ctx()
    );
}

template <class Context>
void BiasAddOp<Context>::RunOnDevice() {
    if (data_format() == "NCHW") {
        outer_dim_ = X(0).dim(0);
        axis_dim_ = X(0).dim(1);
        inner_dim_ = X(0).count(2);
    } else if (data_format() == "NHWC") {
        outer_dim_ = X(0).dim(0);
        axis_dim_ = X(0).dim(-1);
        inner_dim_ = X(0).count(1) / axis_dim_;
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }

    Y(0)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(0));
}

template <class Context> template <typename T>
void BiasAddGradientOp<Context>::RunImpl() {
    if (Y(1)->name() != "NULL") {
        vec32_t dims, axes;
        if (data_format() == "NCHW") {
            dims = {
                (int)outer_dim_,
                (int)axis_dim_,
                (int)inner_dim_,
            }, axes = { 0, 2 };
        } else if (data_format() == "NHWC") {
            dims = {
                (int)outer_dim_,
                (int)inner_dim_,
                (int)axis_dim_,
            }, axes = { 0, 1 };
        }
        kernel::ReduceSum(
            3, dims.data(),
            2, axes.data(),
            1.f,
            X(-1).template data<T, Context>(),
            Y(1)->template mutable_data<T, Context>(),
            ctx()
        );
    }

    if (Y(0)->name() != "NULL" &&
        Y(0)->name() != X(-1).name()) {
        Y(0)->ReshapeLike(X(-1))
            ->CopyFrom(X(-1), ctx());
    }
}

template <class Context>
void BiasAddGradientOp<Context>::RunOnDevice() {
    if (data_format() == "NCHW") {
        outer_dim_ = X(-1).dim(0);
        axis_dim_ = X(-1).dim(1);
        inner_dim_ = X(-1).count(2);
    } else if (data_format() == "NHWC") {
        outer_dim_ = X(-1).dim(0);
        axis_dim_ = X(-1).dim(-1);
        inner_dim_ = X(-1).count(1) / axis_dim_;
    } else {
        LOG(FATAL) << "Unknown DataFormat: " << data_format();
    }

    Y(1)->ReshapeLike(X(0));

    DispatchHelper<TensorTypes
        <float>>::Call(this, X(-1));
}

DEPLOY_CPU(BiasAdd);
#ifdef WITH_CUDA
DEPLOY_CUDA(BiasAdd);
#endif

DEPLOY_CPU(BiasAddGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BiasAddGradient);
#endif

OPERATOR_SCHEMA(BiasAdd)
     /* X, B */
    .NumInputs(2)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

OPERATOR_SCHEMA(BiasAddGradient)
     /* B, dY */
    .NumInputs(2)
     /* dX, dB */
    .NumOutputs(2)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(1), GO(0) }),
            vector<string>({ GI(0), GI(1) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(BiasAdd, GradientMaker);

}  // namespace dragon