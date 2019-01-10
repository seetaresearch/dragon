#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "operators/vision/bias_add_op.h"

namespace dragon {

template <class Context> template <typename T>
void BiasAddOp<Context>::RunWithType() {
    TENSOR_FILL(Input(1), vector<int64_t>(1, dim));
    DECLARE_MULTIPLIER(multiplier, inner_dim);

    auto* Bdata = Input(1).template data<T, Context>();
    auto* Ydata = Output(0)->template mutable_data<T, Context>();

    // Copy X to Y firstly if necessary
    Output(0)->template CopyFrom<Context>(Input(0), ctx());

    kernel::BiasAdd(Output(0)->count(), outer_dim, dim, inner_dim,
        data_format, Bdata, multiplier, Ydata, ctx());
}

template <class Context>
void BiasAddOp<Context>::RunOnDevice() {
    if (data_format == "NCHW") {
        outer_dim = Input(0).dim(0);
        dim = Input(0).dim(1);
        inner_dim = Input(0).count(2);
    } else if (data_format == "NHWC") {
        outer_dim = Input(0).dim(0);
        dim = Input(0).dim(-1);
        inner_dim = Input(0).count(1) / dim;
    } else LOG(FATAL) << "Unknown data format: " << data_format;

    Output(0)->ReshapeLike(Input(0));

    if (XIsType(Input(0), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
}

DEPLOY_CPU(BiasAdd);
#ifdef WITH_CUDA
DEPLOY_CUDA(BiasAdd);
#endif
OPERATOR_SCHEMA(BiasAdd)
    .NumInputs(2).NumOutputs(1)
    .Inplace({ { 0, 0 } });

template <class Context> template <typename T>
void BiasAddGradientOp<Context>::RunWithType() {
    if (Output(1)->name() != "ignore") {
        DECLARE_MULTIPLIER(multiplier, inner_dim);
        auto* dYdata = Input(-1).template data<T, Context>();
        auto* dBias = Output(1)->template mutable_data<T, Context>();
        const int y_offset = dim * inner_dim;
        for (int n = 0; n < outer_dim; n++) {
            if (data_format == "NCHW") {
                math::Gemv(
                    CblasNoTrans, dim, inner_dim,
                        1.f, dYdata, multiplier,
                            1.f, dBias, ctx());
            } else if (data_format == "NHWC") {
                math::Gemv(
                    CblasTrans, inner_dim, dim,
                        1.f, dYdata, multiplier,
                            1.f, dBias, ctx());
            }
            dYdata += y_offset;
        }
    }

    if (Output(0)->name() != "ignore" &&
        Output(0)->name() != Input(-1).name()) {
        Output(0)->ReshapeLike(Input(-1));
        Output(0)->template CopyFrom<Context>(Input(-1), ctx());
    }
}

template <class Context>
void BiasAddGradientOp<Context>::RunOnDevice() {
    if (data_format == "NCHW") {
        outer_dim = Input(-1).dim(0);
        dim = Input(-1).dim(1);
        inner_dim = Input(-1).count(2);
    } else if (data_format == "NHWC") {
        outer_dim = Input(-1).dim(0);
        dim = Input(-1).dim(-1);
        inner_dim = Input(-1).count(1) / dim;
    } else LOG(FATAL) << "Unknown data format: " << data_format;

    Output(1)->ReshapeLike(Input(0));

    if (XIsType(Input(-1), float)) RunWithType<float>();
    else LOG(FATAL) << DTypeHelper(Input(-1), { "float32" });
}

DEPLOY_CPU(BiasAddGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BiasAddGradient);
#endif

OPERATOR_SCHEMA(BiasAddGradient)
    .NumInputs(2).NumOutputs(2)
    .Inplace({ { 1, 0 } });

class GetBiasAddGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetBiasAddGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(1), GO(0) }),
            vector<string>({ GI(0), GI(1) }));
    }
};

REGISTER_GRADIENT(BiasAdd, GetBiasAddGradient);

}  // namespace dragon