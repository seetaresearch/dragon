#include "operators/vision/lrn_op.h"
#include "operators/arithmetic/pow_op.h"
#include "operators/arithmetic/eltwise_op.h"
#include "operators/vision/pooling_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context> template <typename T>
void LRNOp<Context>::AcrossRunWithType() {
    LOG(FATAL) << "lrn with across channels is not implemented,"
               << "\n please compile cuDNN for this module.";
}

template <class Context> template <typename T>
void LRNOp<Context>::SplitRunWithType() {
    sqr_in = ws()->CreateTensor("_t_" + anchor() + "_sqr_in");
    sqr_in->ReshapeLike(input(0));
    sqr_in->Share(input(0));

    prod_in = ws()->CreateTensor("_t_" + anchor() + "_prod_in");
    prod_in->ReshapeLike(input(0));
    prod_in->Share(input(0));
}

template <class Context> template <typename T>
void LRNOp<Context>::SquareRunWithType() {
    sqr_out = ws()->CreateTensor("_t_" + anchor() + "_sqr_out");
    if (!sqr_op) {
        Argument power;
        power.set_name("power"); power.set_f(2.0);
        OperatorDef sqr_op_def = MakeOperatorDef("Pow", "",
                                                 vector<string>({ sqr_in->name() }),
                                                 vector<string>({ sqr_out->name() }),
                                                 vector<Argument>({ power }));
        if (this->op_def().has_device_option())
            sqr_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
        sqr_op.reset(CreateOperator(sqr_op_def, ws()));
    }
    sqr_op->Run();
}

template <class Context> template <typename T>
void LRNOp<Context>::PoolRunWithType() {
    pool_out = ws()->CreateTensor("_t_" + anchor() + "_pool_out");
    if (!pool_op) {
        Argument ks, s, p, mode;
        ks.set_name("kernel_size"); ks.add_ints(local_size);
        s.set_name("stride"); s.add_ints(1);
        p.set_name("pad"); p.add_ints((local_size - 1) / 2);
        mode.set_name("mode"); mode.set_i(AVG_POOLING);
        OperatorDef pool_op_def = MakeOperatorDef("Pooling", "",
                                                  vector<string>({ sqr_out->name() }),
                                                  vector<string>({ pool_out->name() }), 
                                                  vector<Argument>({ ks, s, p, mode }));
        if (this->op_def().has_device_option())
            pool_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
        pool_op.reset(CreateOperator(pool_op_def, ws()));
    }
    pool_op->Run();
}

template <class Context> template <typename T>
void LRNOp<Context>::PowRunWithType() {
    pow_out = ws()->CreateTensor("_t_" + anchor() + "_pow_out");
    if (!pow_op) {
        Argument scale, shift, power;
        scale.set_name("scale"); scale.set_f(alpha);
        shift.set_name("shift"); shift.set_f(1.0);
        power.set_name("power"); power.set_f(-beta);
        OperatorDef pow_op_def = MakeOperatorDef("Pow", "",
                                                 vector<string>({ pool_out->name() }),
                                                 vector<string>({ pow_out->name() }), 
                                                 vector<Argument>({ scale, shift, power }));
        if (this->op_def().has_device_option())
            pow_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
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
                                                  vector<string>({ prod_in->name(), 
                                                                   pow_out->name() }),
                                                  vector<string>({ output(0)->name() }),
                                                  vector<Argument>({ operation }));
        if (this->op_def().has_device_option())
            prod_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
        prod_op.reset(CreateOperator(prod_op_def, ws()));
    }
    prod_op->Run();
}

template <class Context>
void LRNOp<Context>::RunOnDevice() {
    if (mode == ACROSS_CHANNELS) {
        if (input(0).template IsType<float>()) {
            AcrossRunWithType<float>();
        } else { LOG(FATAL) << "unsupported input types."; }
    } 
    else {
        if (input(0).template IsType<float>()) {
            SplitRunWithType<float>();
            SquareRunWithType<float>();
            PoolRunWithType<float>();
            PowRunWithType<float>();
            ProdRunWithType<float>();
        } else { LOG(FATAL) << "unsupported input types."; }
    }
}

DEPLOY_CPU(LRN);
#ifdef WITH_CUDA
DEPLOY_CUDA(LRN);
#endif
OPERATOR_SCHEMA(LRN).NumInputs(1).NumOutputs(1);

template <class Context> template <typename T>
void LRNGradientOp<Context>::AcrossRunWithType() {
    LOG(FATAL) << "lrn with across channels is not implemented,"
               << "\n please compile cuDNN for this module.";
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::ProdRunWithType() {
    prod_in = ws()->GetTensor("_t_" + anchor() + "_prod_in");
    pow_out = ws()->GetTensor("_t_" + anchor() + "_pow_out");
    if (!prod_op) {
        Argument operation;
        operation.set_name("operation"); operation.set_s("PROD");
        OperatorDef prod_op_def = MakeOperatorDef("EltwiseGradient", "",
                                                  vector<string>({ prod_in->name(), 
                                                                   pow_out->name(), 
                                                                   input(-1).name() }),
                                                  vector<string>({ prod_in->name() + "_grad", 
                                                                   pow_out->name() + "_grad" }),
                                                  vector<Argument>({ operation }));
        if (this->op_def().has_device_option())
            prod_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
        prod_op.reset(CreateOperator(prod_op_def, ws()));
    }
    prod_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::PowRunWithType() {
    pool_out = ws()->GetTensor("_t_" + anchor() + "_pool_out");
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
        if (this->op_def().has_device_option())
            pow_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
        pow_op.reset(CreateOperator(pow_op_def, ws()));
    }
    pow_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::PoolRunWithType() {
    sqr_out = ws()->GetTensor("_t_" + anchor() + "_sqr_out");
    if (!pool_op) {
        Argument ks, s, p, mode;
        ks.set_name("kernel_size"); ks.add_ints(local_size);
        s.set_name("stride"); s.add_ints(1);
        p.set_name("pad"); p.add_ints((local_size - 1) / 2);
        mode.set_name("mode"); mode.set_i(AVG_POOLING);
        OperatorDef pool_op_def = MakeOperatorDef("PoolingGradient", "",
                                                  vector<string>({ sqr_out->name(), 
                                                                   pool_out->name(),
                                                                   pool_out->name() + "_grad" }),
                                                  vector<string>({ sqr_out->name() + "_grad" }),
                                                  vector<Argument>({ ks, s, p, mode }));
        if (this->op_def().has_device_option())
            pool_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
        pool_op.reset(CreateOperator(pool_op_def, ws()));
    }
    pool_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::SquareRunWithType() {
    sqr_in = ws()->GetTensor("_t_" + anchor() + "_sqr_in");
    if (!sqr_op) {
        Argument power;
        power.set_name("power"); power.set_f(2.0);
        OperatorDef sqr_op_def = MakeOperatorDef("PowGradient", "",
                                                 vector<string>({ sqr_in->name(), 
                                                                  sqr_out->name(), 
                                                                  sqr_out->name() + "_grad" }),
                                                 vector<string>({ sqr_in->name() + "_grad" }),
                                                 vector<Argument>({ power }));
        if (this->op_def().has_device_option())
            sqr_op_def.mutable_device_option()->CopyFrom(this->op_def().device_option());
        sqr_op.reset(CreateOperator(sqr_op_def, ws()));
    }
    sqr_op->Run();
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::SplitRunWithType() {
    Tensor* g_sqr_in = ws()->GetTensor(sqr_in->name() + "_grad");
    Tensor* g_prod_in = ws()->GetTensor(prod_in->name() + "_grad");
    output(0)->ReshapeLike(input(0));

    auto* data0 = g_sqr_in->template data<T, Context>();
    auto* data1 = g_prod_in->template data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    math::Add<T, Context>(output(0)->count(), data0, data1, dXdata);
}

template <class Context>
void LRNGradientOp<Context>::RunOnDevice() {
    if (mode == ACROSS_CHANNELS) {
        if (input(0).template IsType<float>()) {
            AcrossRunWithType<float>();
        } else { LOG(FATAL) << "unsupported input types."; }
    } 
    else {
        if (input(0).template IsType<float>()) {
            ProdRunWithType<float>();
            PowRunWithType<float>();
            PoolRunWithType<float>();
            SquareRunWithType<float>();
            SplitRunWithType<float>();
        } else { LOG(FATAL) << "unsupported input types."; }
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