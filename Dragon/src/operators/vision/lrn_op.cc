#include "core/workspace.h"
#include "utils/math_functions.h"
#include "operators/arithmetic/pow_op.h"
#include "operators/arithmetic/eltwise_op.h"
#include "operators/vision/lrn_op.h"
#include "operators/vision/pool_op.h"

namespace dragon {

template <class Context> template <typename T>
void LRNOp<Context>::AcrossRunImpl() {
    LOG(FATAL) << "Across Channels is not implemented."
               << "\nCompile CuDNN for this module.";
}

template <class Context> template <typename T>
void LRNOp<Context>::SplitRunImpl() {
    sqr_in_ = ws()
        ->CreateTensor(unique_name("sqr/in"))
        ->ReshapeLike(X(0))
        ->CopyFrom(X(0), ctx());
    prod_in_ = ws()
        ->CreateTensor(unique_name("prod/in"))
        ->ReshapeLike(X(0))
        ->CopyFrom(X(0), ctx());
}

template <class Context> template <typename T>
void LRNOp<Context>::SquareRunImpl() {
    sqr_out_ = ws()->CreateTensor(unique_name("sqr/out"));
    if (!sqr_op_) {
        Argument power;
        power.set_name("power"); power.set_f(2.f);
        auto sqr_op_def = MakeOperatorDef(
            "Pow", "",
            vector<string>({ sqr_in_->name() }),
            vector<string>({ sqr_out_->name() }),
            vector<Argument>({ power })
        );
        if (def().has_device_option())
            sqr_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        sqr_op_.reset(NewOperator(sqr_op_def, ws()));
    }
    sqr_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void LRNOp<Context>::PoolRunImpl() {
    pool_out_ = ws()->CreateTensor(unique_name("pool/out"));
    if (!pool_op_) {
        Argument ks, s, p, m, df;
        ks.set_name("kernel_size"); ks.add_ints(local_size_);
        s.set_name("stride"); s.add_ints(1);
        p.set_name("pad"); p.add_ints((local_size_ - 1) / 2);
        m.set_name("mode"); m.set_s("AVG");
        df.set_name("data_format"); df.set_s(data_format());
        auto pool_op_def = MakeOperatorDef(
            "Pool2d", "",
            vector<string>({ sqr_out_->name() }),
            vector<string>({ pool_out_->name() }),
            vector<Argument>({ ks, s, p, m, df })
        );
        if (def().has_device_option())
            pool_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pool_op_.reset(NewOperator(pool_op_def, ws()));
    }
    pool_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void LRNOp<Context>::PowRunImpl() {
    pow_out_ = ws()->CreateTensor(unique_name("pow/out"));
    if (!pow_op_) {
        Argument scale, shift, power;
        scale.set_name("scale"); scale.set_f(alpha_);
        shift.set_name("shift"); shift.set_f(1.f);
        power.set_name("power"); power.set_f(-beta_);
        auto pow_op_def = MakeOperatorDef(
            "Pow", "",
            vector<string>({ pool_out_->name() }),
            vector<string>({ pow_out_->name() }),
            vector<Argument>({ scale, shift, power })
        );
        if (def().has_device_option())
            pow_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pow_op_.reset(NewOperator(pow_op_def, ws()));
    }
    pow_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void LRNOp<Context>::ProdRunImpl() {
    if (!prod_op_) {
        Argument operation;
        operation.set_name("operation"); operation.set_s("PROD");
        auto prod_op_def = MakeOperatorDef(
            "Eltwise", "",
            vector<string>({ prod_in_->name(), pow_out_->name() }),
            vector<string>({ Y(0)->name() }),
            vector<Argument>({ operation })
        );
        if (def().has_device_option())
            prod_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        prod_op_.reset(NewOperator(prod_op_def, ws()));
    }
    prod_op_->Run(ctx()->stream_id());
}

template <class Context>
void LRNOp<Context>::RunOnDevice() {
    if (mode_ == "ACROSS_CHANNELS") {
        if (XIsType(X(0), float)) {
            AcrossRunImpl<float>();
        } else LOG(FATAL) << DTypeString(X(0), { "float32" });
    } else if (mode_ == "WITHIN_CHANNEL") {
        if (XIsType(X(0), float)) {
            SplitRunImpl<float>();
            SquareRunImpl<float>();
            PoolRunImpl<float>();
            PowRunImpl<float>();
            ProdRunImpl<float>();
        } else LOG(FATAL) << DTypeString(X(0), { "float32" });
    } else {
        LOG(FATAL) << "Unknown Mode: " << mode_;
    }
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::AcrossRunImpl() {
    LOG(FATAL) << "Across Channels is not implemented,"
               << "\nCompile cuDNN for this module.";
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::ProdRunImpl() {
    prod_in_ = ws()->GetTensor(unique_name("prod/in"));
    pow_out_ = ws()->GetTensor(unique_name("pow/out"));
    if (!prod_op_) {
        Argument operation;
        operation.set_name("operation"); operation.set_s("PROD");
        auto prod_op_def = MakeOperatorDef(
            "EltwiseGradient", "",
            vector<string>({ prod_in_->name(),
                             pow_out_->name(),
                             X(-1).name() }),
            vector<string>({ prod_in_->name() + "_grad",
                             pow_out_->name() + "_grad" }),
            vector<Argument>({ operation }));
        if (def().has_device_option())
            prod_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        prod_op_.reset(NewOperator(prod_op_def, ws()));
    }
    prod_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::PowRunImpl() {
    pool_out_ = ws()->GetTensor(unique_name("pool/out"));
    if (!pow_op_) {
        Argument scale, shift, power;
        scale.set_name("scale"); scale.set_f(alpha_);
        shift.set_name("shift"); shift.set_f(1.f);
        power.set_name("power"); power.set_f(-beta_);
        auto pow_op_def = MakeOperatorDef(
            "PowGradient", "",
            vector<string>({ pool_out_->name(),
                             pow_out_->name(),
                             pow_out_->name() + "_grad" }),
            vector<string>({ pool_out_->name() + "_grad" }),
            vector<Argument>({ scale, shift, power }));
        if (def().has_device_option())
            pow_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pow_op_.reset(NewOperator(pow_op_def, ws()));
    }
    pow_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::PoolRunImpl() {
    sqr_out_ = ws()->GetTensor(unique_name("sqr/out"));
    if (!pool_op_) {
        Argument ks, s, p, m, df;
        ks.set_name("kernel_size"); ks.add_ints(local_size_);
        s.set_name("stride"); s.add_ints(1);
        p.set_name("pad"); p.add_ints((local_size_ - 1) / 2);
        m.set_name("mode"); m.set_s("AVG");
        df.set_name("data_format"); df.set_s(data_format());
        auto pool_op_def = MakeOperatorDef(
            "Pool2dGradient", "",
            vector<string>({ sqr_out_->name(),
                             pool_out_->name(),
                             pool_out_->name() + "_grad" }),
            vector<string>({ sqr_out_->name() + "_grad" }),
            vector<Argument>({ ks, s, p, m, df }));
        if (def().has_device_option())
            pool_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        pool_op_.reset(NewOperator(pool_op_def, ws()));
    }
    pool_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::SquareRunImpl() {
    sqr_in_ = ws()->GetTensor(unique_name("sqr/in"));
    if (!sqr_op_) {
        Argument power;
        power.set_name("power"); power.set_f(2.f);
        auto sqr_op_def = MakeOperatorDef(
            "PowGradient", "",
            vector<string>({ sqr_in_->name(),
                             sqr_out_->name(),
                             sqr_out_->name() + "_grad" }),
            vector<string>({ sqr_in_->name() + "_grad" }),
            vector<Argument>({ power }));
        if (def().has_device_option())
            sqr_op_def.mutable_device_option()
                ->CopyFrom(def().device_option());
        sqr_op_.reset(NewOperator(sqr_op_def, ws()));
    }
    sqr_op_->Run(ctx()->stream_id());
}

template <class Context> template <typename T>
void LRNGradientOp<Context>::SplitRunImpl() {
    auto* dy1 = ws()
        ->GetTensor(sqr_in_->name() + "_grad")
        ->template data<T, Context>();
    auto * dy2 = ws()
        ->GetTensor(prod_in_->name() + "_grad")
        ->template data<T, Context>();

    Y(0)->ReshapeLike(X(0));

    auto* dx = Y(0)->template mutable_data<T, Context>();
    math::Add(Y(0)->count(), dy1, dy2, dx, ctx());
}

template <class Context>
void LRNGradientOp<Context>::RunOnDevice() {
    if (mode_ == "ACROSS_CHANNELS") {
        if (XIsType(X(0), float)) {
            AcrossRunImpl<float>();
        } else {
            LOG(FATAL) << DTypeString(
                X(0), { "float32" }
            );
        }
    } else if (mode_ == "WITHIN_CHANNEL") {
        if (XIsType(X(0), float)) {
            ProdRunImpl<float>();
            PowRunImpl<float>();
            PoolRunImpl<float>();
            SquareRunImpl<float>();
            SplitRunImpl<float>();
        } else {
            LOG(FATAL) << DTypeString(
                X(0), { "float32" }
            );
        }
    } else {
        LOG(FATAL) << "Unknown Mode: " << mode_;
    }
}

DEPLOY_CPU(LRN);
#ifdef WITH_CUDA
DEPLOY_CUDA(LRN);
#endif

DEPLOY_CPU(LRNGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(LRNGradient);
#endif

OPERATOR_SCHEMA(LRN)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(LRNGradient)
     /* X, Y, dY */
    .NumInputs(3)
     /* dX */
    .NumOutputs(1);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override{
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), O(0), GO(0) }),
            vector<string>({ GI(0) }));
    }
};

}  // namespace

REGISTER_GRADIENT(LRN, GradientMaker);

}  // namespace dragon