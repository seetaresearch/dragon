#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/pow_op.h"
#include "operators/arithmetic/eltwise_op.h"
#include "operators/vision/lrn_op.h"
#include "operators/vision/pooling_op.h"

namespace dragon {

template <class Context> template <typename T>
void LRNOp<Context>::AcrossRunWithType() {
    LOG(FATAL) << "Across Channels is not implemented."
               << "\nCompile cuDNN for this module.";
}

template <class Context> template <typename T>
void LRNOp<Context>::SplitRunWithType() {
    sqr_in = ws()->CreateTensor("/mnt/" + anchor() + "/sqr/in");
    sqr_in->ReshapeLike(Input(0));
    sqr_in->template CopyFrom<Context>(Input(0));

    prod_in = ws()->CreateTensor("/mnt/" + anchor() + "/prod/in");
    prod_in->ReshapeLike(Input(0));
    prod_in->template CopyFrom<Context>(Input(0));
}

template <class Context> template <typename T>
void LRNOp<Context>::SquareRunWithType() {
    sqr_out = ws()->CreateTensor("/mnt/" + anchor() + "/sqr/out");
    if (!sqr_op) {
        Argument power;
        power.set_name("power"); power.set_f(2.0);
        OperatorDef sqr_op_def = MakeOperatorDef("Pow", "",
            vector<string>({ sqr_in->name() }),
                vector<string>({ sqr_out->name() }),
                    vector<Argument>({ power }));
        if (def().has_device_option())
            sqr_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        sqr_op.reset(CreateOperator(sqr_op_def, ws()));
    }
    sqr_op->Run();
}

template <class Context> template <typename T>
void LRNOp<Context>::PoolRunWithType() {
    pool_out = ws()->CreateTensor("/mnt/" + anchor() + "/pool/out");
    if (!pool_op) {
        Argument ks, s, p, m, df;
        ks.set_name("kernel_size"); ks.add_ints(local_size);
        s.set_name("stride"); s.add_ints(1);
        p.set_name("pad"); p.add_ints((local_size - 1) / 2);
        m.set_name("mode"); m.set_s("AVG");
        df.set_name("data_format"); df.set_s(data_format);
        OperatorDef pool_op_def = MakeOperatorDef("Pooling2d", "",
            vector<string>({ sqr_out->name() }),
                vector<string>({ pool_out->name() }),
                    vector<Argument>({ ks, s, p, m, df }));
        if (def().has_device_option())
            pool_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pool_op.reset(CreateOperator(pool_op_def, ws()));
    }
    pool_op->Run();
}

template <class Context> template <typename T>
void LRNOp<Context>::PowRunWithType() {
    pow_out = ws()->CreateTensor("/mnt/" + anchor() + "/pow/out");
    if (!pow_op) {
        Argument scale, shift, power;
        scale.set_name("scale"); scale.set_f(alpha);
        shift.set_name("shift"); shift.set_f(1.0);
        power.set_name("power"); power.set_f(-beta);
        OperatorDef pow_op_def = MakeOperatorDef("Pow", "",
            vector<string>({ pool_out->name() }),
                vector<string>({ pow_out->name() }),
                    vector<Argument>({ scale, shift, power }));
        if (def().has_device_option())
            pow_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pow_op.reset(CreateOperator(pow_op_def, ws()));
    }
    pow_op->Run();
}

template <class Context> template <typename T>
void LRNOp<Context>::ProdRunWithType() {
    if (!prod_op) {
        Argument operation;
        operation.set_name("operation"); operation.set_s("PROD");
        OperatorDef prod_op_def = MakeOperatorDef("Eltwise", "",
            vector<string>({ prod_in->name(), pow_out->name() }),
                vector<string>({ Output(0)->name() }),
                    vector<Argument>({ operation }));
        if (def().has_device_option())
            prod_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        prod_op.reset(CreateOperator(prod_op_def, ws()));
    }
    prod_op->Run();
}

template <class Context>
void LRNOp<Context>::RunOnDevice() {
    if (mode == "ACROSS_CHANNELS") {
        if (XIsType(Input(0), float)) {
            AcrossRunWithType<float>();
        } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } else if (mode == "WITHIN_CHANNEL") {
        if (XIsType(Input(0), float)) {
            SplitRunWithType<float>();
            SquareRunWithType<float>();
            PoolRunWithType<float>();
            PowRunWithType<float>();
            ProdRunWithType<float>();
        } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } else {
        LOG(FATAL) << "Unsupported lrn mode: " << mode;
    }
}

DEPLOY_CPU(LRN);
#ifdef WITH_CUDA
DEPLOY_CUDA(LRN);
#endif
OPERATOR_SCHEMA(LRN).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void LRNGradientOp<Context>::AcrossRunWithType() {
    LOG(FATAL) << "Across Channels is not implemented,"
               << "\nCompile cuDNN for this module.";
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::ProdRunWithType() {
    prod_in = ws()->GetTensor("/mnt/" + anchor() + "/prod/in");
    pow_out = ws()->GetTensor("/mnt/" + anchor() + "/pow/out");
    if (!prod_op) {
        Argument operation;
        operation.set_name("operation"); operation.set_s("PROD");
        OperatorDef prod_op_def = MakeOperatorDef("EltwiseGradient", "",
            vector<string>({ prod_in->name(),
                                 pow_out->name(),
                                     Input(-1).name() }),
                vector<string>({ prod_in->name() + "_grad",
                                     pow_out->name() + "_grad" }),
                    vector<Argument>({ operation }));
        if (def().has_device_option())
            prod_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        prod_op.reset(CreateOperator(prod_op_def, ws()));
    }
    prod_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::PowRunWithType() {
    pool_out = ws()->GetTensor("/mnt/" + anchor() + "/pool/out");
    if (!pow_op) {
        Argument scale, shift, power;
        scale.set_name("scale"); scale.set_f(alpha);
        shift.set_name("shift"); shift.set_f(1.0);
        power.set_name("power"); power.set_f(-beta);
        OperatorDef pow_op_def = MakeOperatorDef("PowGradient", "",
            vector<string>({ pool_out->name(),
                                pow_out->name(),
                                    pow_out->name() + "_grad" }),
                vector<string>({ pool_out->name() + "_grad" }),
                    vector<Argument>({ scale, shift, power }));
        if (def().has_device_option())
            pow_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pow_op.reset(CreateOperator(pow_op_def, ws()));
    }
    pow_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::PoolRunWithType() {
    sqr_out = ws()->GetTensor("/mnt/" + anchor() + "/sqr/out");
    if (!pool_op) {
        Argument ks, s, p, m, df;
        ks.set_name("kernel_size"); ks.add_ints(local_size);
        s.set_name("stride"); s.add_ints(1);
        p.set_name("pad"); p.add_ints((local_size - 1) / 2);
        m.set_name("mode"); m.set_s("AVG");
        df.set_name("data_format"); df.set_s(data_format);
        OperatorDef pool_op_def = MakeOperatorDef("Pooling2dGradient", "",
            vector<string>({ sqr_out->name(),
                                 pool_out->name(),
                                     pool_out->name() + "_grad" }),
                vector<string>({ sqr_out->name() + "_grad" }),
                    vector<Argument>({ ks, s, p, m, df }));
        if (def().has_device_option())
            pool_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pool_op.reset(CreateOperator(pool_op_def, ws()));
    }
    pool_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::SquareRunWithType() {
    sqr_in = ws()->GetTensor("/mnt/" + anchor() + "/sqr/in");
    if (!sqr_op) {
        Argument power;
        power.set_name("power"); power.set_f(2.0);
        OperatorDef sqr_op_def = MakeOperatorDef("PowGradient", "",
            vector<string>({ sqr_in->name(),
                                 sqr_out->name(),
                                     sqr_out->name() + "_grad" }),
                vector<string>({ sqr_in->name() + "_grad" }),
                    vector<Argument>({ power }));
        if (def().has_device_option())
            sqr_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        sqr_op.reset(CreateOperator(sqr_op_def, ws()));
    }
    sqr_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::SplitRunWithType() {
    Tensor* g_sqr_in = ws()->GetTensor(sqr_in->name() + "_grad");
    Tensor* g_prod_in = ws()->GetTensor(prod_in->name() + "_grad");
    Output(0)->ReshapeLike(Input(0));

    auto* data0 = g_sqr_in->template data<T, Context>();
    auto* data1 = g_prod_in->template data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    math::Add<T, Context>(Output(0)->count(), data0, data1, dXdata);
}

template <class Context>
void LRNGradientOp<Context>::RunOnDevice() {
    if (mode == "ACROSS_CHANNELS") {
        if (XIsType(Input(0), float)) AcrossRunWithType<float>();
        else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } else if (mode == "WITHIN_CHANNEL") {
        if (XIsType(Input(0), float)) {
            ProdRunWithType<float>();
            PowRunWithType<float>();
            PoolRunWithType<float>();
            SquareRunWithType<float>();
            SplitRunWithType<float>();
        } else LOG(FATAL) << DTypeHelper(Input(0), { "float32" });
    } else {
        LOG(FATAL) << "Unsupported lrn mode: " << mode;
    }
}

DEPLOY_CPU(LRNGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LRNGradient);
#endif
OPERATOR_SCHEMA(LRNGradient).NumInputs(3).NumOutputs(1);

class GetLRNGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetLRNGradient);
    vector<OperatorDef> MakeDefs() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), O(0), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(LRN, GetLRNGradient);

}    // namespace dragon