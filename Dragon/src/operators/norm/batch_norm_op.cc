#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/norm/batch_norm_op.h"

namespace dragon {

template <class Context>
template <typename Tx, typename Tp>
void BatchNormOp<Context>::TrainingImpl() {
    TENSOR_FILL_WITH_TYPE(X(1), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(2), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(3), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(4), vec64_t({ C_ }), Tp);

    auto* x = X(0).template data<Tx, Context>();
    auto* rm = X(1).template mutable_data<Tp, Context>();
    auto* rv = X(2).template mutable_data<Tp, Context>();
    auto* gamma = X(3).template data<Tp, Context>();
    auto* beta = X(4).template data<Tp, Context>();
    auto* mu = mean_->template mutable_data<Tp, Context>();
    auto* rsig = var_->template mutable_data<Tp, Context>();
    auto* s = scale_.template mutable_data<Tp, Context>();
    auto* b = bias_.template mutable_data<Tp, Context>();
    auto* y = Y(0)->template mutable_data<Tx, Context>();

    // Compute the moments
    if (data_format() == "NCHW") {
        const std::array<int, 3> dims = { (int)N_, (int)C_, (int)S_ };
        const std::array<int, 3> axes = { 0, 2 };
        kernel::Moments(
            3, dims.data(),
            2, axes.data(),
            x, mu, rsig, ctx()
        );
    } else if (data_format() == "NHWC") {
        const std::array<int, 2> dims = { (int)(N_ * S_), (int)C_ };
        const std::array<int, 1> axes = { 0 };
        kernel::Moments(
            2, dims.data(),
            1, axes.data(),
            x, mu, rsig, ctx()
        );
    }

    // Compute moving average
    if (!is_recomp_) {
        // Running(X)
        //   = (1 - momentum) * Cur(X)
        //     + momentum * Running(X)
        math::Axpby<Tp, Context>(
            (int)C_,
            1.f - momentum_, mu,
            momentum_, rm, ctx()
        );
        math::Axpby<Tp, Context>(
            (int)C_,
            1.f - momentum_, rsig,
            momentum_, rv, ctx()
        );
    }

    // Fuse: [mu, rsig, alpha, beta] => [scale, bias]
    math::InvStd(C_, eps_, rsig, rsig, ctx());
    math::Mul(C_, gamma, rsig, s, ctx());
    math::Mul(C_, s, mu, b, ctx());
    math::Sub(C_, beta, b, b, ctx());

    // Affine
    if (data_format() == "NCHW") {
        kernel::Affine(
            N_, C_, S_,
            x, s, b,
            y, ctx()
        );
    } else if (data_format() == "NHWC") {
        kernel::Affine(
            N_ * S_, C_, 1,
            x, s, b,
            y, ctx()
        );
    }
}

template <class Context>
template <typename Tx, typename Tp>
void BatchNormOp<Context>::InferenceImpl() {
    TENSOR_FILL_WITH_TYPE(X(1), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(2), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(3), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(4), vec64_t({ C_ }), Tp);

    auto* x = X(0).template data<Tx, Context>();
    auto* rm = X(1).template data<Tp, Context>();
    auto* rv = X(2).template data<Tp, Context>();
    auto* gamma = X(3).template data<Tp, Context>();
    auto* beta = X(4).template data<Tp, Context>();
    auto* s = scale_.template mutable_data<Tp, Context>();
    auto* b = bias_.template mutable_data<Tp, Context>();
    auto* y = Y(0)->template mutable_data<Tx, Context>();

    // Fuse: [rmean, rvar, alpha, beta]
    //    => [scale, bias]
    math::InvStd(C_, eps_, rv, b, ctx());
    math::Mul(C_, gamma, b, s, ctx());
    math::Mul(C_, s, rm, b, ctx());
    math::Sub(C_, beta, b, b, ctx());

    // Affine
    if (data_format() == "NCHW") {
        kernel::Affine(
            N_, C_, S_,
            x, s, b,
            y, ctx()
        );
    } else if (data_format() == "NHWC") {
        kernel::Affine(
            N_ * S_, C_, 1,
            x, s, b,
            y, ctx()
        );
    }
}

template <class Context>
void BatchNormOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats_ == -1) {
        is_training_ = phase() == "TRAIN" ? 1 : 0;
    } else {
        is_training_ = use_stats_ > 0 ? 0 : 1;
    }

    // Get the recomputing flag
    is_recomp_ = ws()
        ->GetTensor("/opt/recomp_flag")
        ->template data<bool, CPUContext>()[0];

    // Determine the data format
    auto axis = axis_;
    this->data_format_ = "NCHW";
    if (axis == -1) axis += X(0).ndim();
    if (axis + 1 == X(0).ndim())
        this->data_format_ = "NHWC";
    N_ = X(0).dim(0);
    C_ = X(0).dim(axis);
    S_ = X(0).count() / N_ / C_;

    // Create the shared resources
    mean_ = ws()
        ->CreateTensor(unique_name("mu"))
        ->Reshape({ C_ });

    var_ = ws()
        ->CreateTensor(unique_name("rsig"))
        ->Reshape({ C_ });

    // Reshape
    scale_.Reshape({ C_ });
    bias_.Reshape({ C_ });
    Y(0)->ReshapeLike(X(0));
}

template <class Context>
void BatchNormOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        if (is_training_) {
            TrainingImpl<float, float>();
        } else {
            InferenceImpl<float, float>();
        }
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

template <class Context>
template <typename Tx, typename Tp>
void BatchNormGradientOp<Context>::TrainingImpl() {
    auto* x = X(0).template data<Tx, Context>();
    auto* mu = mean_->template data<Tp, Context>();
    auto* rsig = var_->template data<Tp, Context>();
    auto* gamma = X(3).template data<Tp, Context>();
    auto* dy = X(-1).template data<Tx, Context>();
    auto* ds = dscale_.template mutable_data<Tp, Context>();
    auto* db = dbias_.template mutable_data<Tp, Context>();
    auto* dx = Y(0)->template mutable_data<Tx, Context>();
    auto* dgamma = Y(1)->template mutable_data<Tp, Context>();
    auto* dbeta = Y(2)->template mutable_data<Tp, Context>();

    kernel::BatchNormBackwardTraining(
        N_, C_, S_,
        data_format(),
        x, mu, rsig, gamma, dy,
        ds, db, dx, dgamma, dbeta, ctx()
    );
}

template <class Context> template <typename Tx, typename Tp>
void BatchNormGradientOp<Context>::InferenceImpl() {
    auto* x = X(0).template data<Tx, Context>();
    auto* rm = X(1).template data<Tp, Context>();
    auto* rv = X(2).template data<Tp, Context>();
    auto* gamma = X(3).template data<Tp, Context>();
    auto* dy = X(-1).template data<Tx, Context>();
    auto* dx = Y(0)->template mutable_data<Tx, Context>();
    auto* rsig = var_->template mutable_data<Tp, Context>();

    Tp* dgamma = nullptr, *dbeta = nullptr;

    // Gradient w.r.t. gamma or beta if necessary
    if (Y(1)->name() != "NULL" ||
        Y(2)->name() != "NULL") {
        dgamma = Y(1)->template mutable_data<Tp, Context>();
        dbeta = Y(2)->template mutable_data<Tp, Context>();
    }

    math::InvStd(C_, eps_, rv, rsig, ctx());

    kernel::BatchNormBackwardInference(
        N_, C_, S_,
        data_format(),
        x, rm, rsig, gamma, dy,
        dx, dgamma, dbeta, ctx()
    );
}

template <class Context>
void BatchNormGradientOp<Context>::Reshape() {
    // Determine the mode
    if (use_stats_ == -1) {
        is_training_ = phase() == "TRAIN" ? 1 : 0;
    } else {
        is_training_ = use_stats_ > 0 ? 0 : 1;
    }

    // Determine the data format
    auto axis = axis_;
    this->data_format_ = "NCHW";
    if (axis == -1) axis += X(0).ndim();
    if (axis + 1 == X(0).ndim())
        this->data_format_ = "NHWC";
    N_ = X(0).dim(0);
    C_ = X(0).dim(axis);
    S_ = X(0).count() / N_ / C_;

    // Get the shared resources
    mean_ = ws()->GetTensor(unique_name("mu"));
    var_ = ws()->GetTensor(unique_name("rsig"));

    // Reshape
    dscale_.Reshape({ C_ });
    dbias_.Reshape({ C_ });
    Y(0)->ReshapeLike(X(0));  // dx
    Y(1)->Reshape({ C_ });    // dgamma
    Y(2)->Reshape({ C_ });    // dbeta
}

template <class Context>
void BatchNormGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        if (is_training_) {
            TrainingImpl<float, float>();
        } else {
            InferenceImpl<float, float>();
        }
    } else {
        LOG(FATAL) << DTypeString(
            X(0), { "float32" }
        );
    }
}

DEPLOY_CPU(BatchNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchNorm);
#endif

DEPLOY_CPU(BatchNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(BatchNormGradient);
#endif

OPERATOR_SCHEMA(BatchNorm)
     /* X, Mean, Var, Gamma, Beta */
    .NumInputs(5)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(BatchNormGradient)
     /* X, Mean, Var, Gamma, dY */
    .NumInputs(5)
     /* dX, dGamma, dBeta */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), I(2), I(3), GO(0) }),
            vector<string>({ GI(0), GI(3), GI(4) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(BatchNorm, GradientMaker);

}  // namespace dragon