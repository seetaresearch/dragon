#include "core/workspace.h"
#include "utils/filler.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/norm/group_norm_op.h"

namespace dragon {

template <class Context>
template <typename Tx, typename Tp>
void GroupNormOp<Context>::RunImpl() {
    TENSOR_FILL_WITH_TYPE(X(1), vec64_t({ C_ }), Tp);
    TENSOR_FILL_WITH_TYPE(X(2), vec64_t({ C_ }), Tp);

    auto* x = X(0).template data<Tx, Context>();
    auto* gamma = X(1).template data<Tp, Context>();
    auto* beta = X(2).template data<Tp, Context>();
    auto* mu = mean_->template mutable_data<Tp, Context>();
    auto* rsig = var_->template mutable_data<Tp, Context>();
    auto* s = scale_.template mutable_data<Tp, Context>();
    auto* b = bias_.template mutable_data<Tp, Context>();
    auto* y = Y(0)->template mutable_data<Tx, Context>();

    // Compute the moments
    if (data_format() == "NCHW") {
        vec32_t dims = { (int)(N_ * G_), (int)(D_ * S_) };
        vec32_t axes = { 1 };
        kernel::Moments(
            2, dims.data(),
            1, axes.data(),
            x, mu, rsig, ctx()
        );
    } else if (data_format() == "NHWC") {
        vec32_t dims = { (int)N_, (int)S_, (int)G_, (int)D_ };
        vec32_t axes = { 1, 3 };
        kernel::Moments(
            4, dims.data(),
            2, axes.data(),
            x, mu, rsig, ctx()
        );
    }

    math::InvStd(N_ * G_, eps_, rsig, rsig, ctx());

    kernel::GroupNormForward(
        N_, G_, D_, S_,
        data_format(),
        x, mu, rsig, gamma, beta,
        s, b, y, ctx()
    );
}

template <class Context>
void GroupNormOp<Context>::Reshape() {
    // Determine the data format
    auto axis = axis_;
    this->data_format_ = "NCHW";
    if (axis == -1) axis += X(0).ndim();
    if (axis + 1 == X(0).ndim())
        this->data_format_ = "NHWC";
    if (X(0).ndim() == 2)
        this->data_format_ = "NCHW";
    N_ = X(0).dim(0);
    C_ = X(0).dim(axis);
    S_ = X(0).count() / N_ / C_;

    // InstanceNorm, LayerNorm or GroupNorm ?
    G_ = group_ > 0 ? group_ : C_; D_ = C_ / G_;

    // Check the channels and groups
    CHECK_EQ(C_ % G_, 0)
        << "\nThe " << C_ << " channels "
        << "can not be split into " << G_ << " groups.";
    if (G_ == C_ && X(0).ndim() == 2)
        LOG(WARNING) << "The 2d input will output all zeros.";

    // Create the shared resources
    mean_ = ws()
        ->CreateTensor(unique_name("mu"))
        ->Reshape({ N_ * G_ });

    var_ = ws()
        ->CreateTensor(unique_name("rsig"))
        ->Reshape({ N_ * G_ });

    // Reshape
    scale_.Reshape({ N_ * C_ });
    bias_.Reshape({ N_ * C_ });
    Y(0)->ReshapeLike(X(0));
}

template <class Context>
void GroupNormOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        RunImpl<float, float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16, float>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template <class Context>
template <typename Tx, typename Tp>
void GroupNormGradientOp<Context>::RunImpl() {
    auto* x = X(0).template data<Tx, Context>();
    auto* mu = mean_->template data<Tp, Context>();
    auto* rsig = var_->template data<Tp, Context>();
    auto* gamma = X(1).template data<Tp, Context>();
    auto* dy = X(-1).template data<Tx, Context>();
    auto* ds = dscale_.template mutable_data<Tp, Context>();
    auto* db = dbias_.template mutable_data<Tp, Context>();
    auto* dx = Y(0)->template mutable_data<Tx, Context>();
    auto* dgamma = Y(1)->template mutable_data<Tp, Context>();
    auto* dbeta = Y(2)->template mutable_data<Tp, Context>();

    kernel::GroupNormBackward(
        N_, G_, D_, S_,
        data_format(),
        x, mu, rsig, gamma, dy,
        ds, db, dx, dgamma, dbeta, ctx()
    );
}

template <class Context>
void GroupNormGradientOp<Context>::Reshape() {
    // Determine the data format
    auto axis = axis_;
    this->data_format_ = "NCHW";
    if (axis == -1) axis += X(0).ndim();
    if (axis + 1 == X(0).ndim())
        this->data_format_ = "NHWC";
    if (X(0).ndim() == 2)
        this->data_format_ = "NCHW";
    N_ = X(0).dim(0);
    C_ = X(0).dim(axis);
    S_ = X(0).count() / N_ / C_;

    // InstanceNorm, LayerNorm or GroupNorm ?
    G_ = group_ > 0 ? group_ : C_; D_ = C_ / G_;

    // Check the channels and groups
    CHECK_EQ(C_ % G_, 0)
        << "\nThe " << C_ << " channels "
        << "can not be split into " << G_ << " groups.";
    if (G_ == C_ && X(0).ndim() == 2)
        LOG(WARNING) << "The 2d input will output all zeros.";

    // Get the shared resources
    mean_ = ws()->GetTensor(unique_name("mu"));
    var_ = ws()->GetTensor(unique_name("rsig"));

    // Reshape
    dscale_.Reshape({ N_ * G_ });
    dbias_.Reshape({ N_ * G_ });
    Y(0)->ReshapeLike(X(0));  // dx
    Y(1)->Reshape({ C_ });    // dgamma
    Y(2)->Reshape({ C_ });    // dbeta
}

template <class Context>
void GroupNormGradientOp<Context>::RunOnDevice() {
    Reshape();

    if (XIsType(X(0), float)) {
        RunImpl<float, float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16, float>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CPU(GroupNorm);
#ifdef WITH_CUDA
DEPLOY_CUDA(GroupNorm);
#endif

DEPLOY_CPU(GroupNormGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(GroupNormGradient);
#endif

OPERATOR_SCHEMA(GroupNorm)
     /* X, Gamma, Beta */
    .NumInputs(3)
     /* Y */
    .NumOutputs(1);

OPERATOR_SCHEMA(GroupNormGradient)
     /* X, Gamma, dY */
    .NumInputs(3)
     /* dX, dGamma, dBeta */
    .NumOutputs(3);

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string>({ I(0), I(1), GO(0) }),
            vector<string>({ GI(0), GI(1), GI(2) })
        );
    }
};

}  // namespace

REGISTER_GRADIENT(GroupNorm, GradientMaker);

}  // namespace dragon