#include "operators/activation/softmax_op.h"
#include "operators/loss/sparse_softmax_focal_loss_op.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "utils/proto_utils.h"

namespace dragon {

template <class Context> template <typename T>
void SparseSoftmaxFocalLossOp<Context>::RunWithType() {
    auto* prob_data = this->prob->template data<T, Context>();
    auto* label_data = input(1).template data<T, Context>();
    auto* loss_data = this->losses.template mutable_data<T, Context>();
    auto* valid_data = this->valid.template mutable_data<T, Context>();
    auto* scale_data = scale->template mutable_data<T, Context>();

    kernel::SparseSoftmaxFocalLoss<T, Context>(input(0).count(), 
                                             input(0).dim(axis),
                                                      outer_dim, 
                                                      inner_dim,
                                                          alpha,
                                                          gamma,
                                                      prob_data, 
                                                     label_data, 
                                                     scale_data,
                                                      loss_data, 
                                                     valid_data, 
                                                 &this->ignore);

    if (normalization == "UNIT") {
        if (use_pseudo_metric) {
            math::MulScalar<T, Context>(this->losses.count(), 
                                                 1.0 / alpha, 
                                                  loss_data);
        }
        output(0)->ReshapeLike(this->losses);
        output(0)->Share(this->losses);
        return;
    }

    T normalizer;
    if (normalization == "VALID")
        normalizer = math::ASum<T, Context>(this->valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = outer_dim;
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(this->losses.count(), loss_data);
    loss = use_pseudo_metric ? loss / alpha : loss;
    output(0)->Reshape(vector<TIndex>(1, 1));
    auto* Ydata = output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = loss / normalizer;
}

template <class Context>
void SparseSoftmaxFocalLossOp<Context>::RunOnDevice() {
    outer_dim = input(0).count(0, axis);
    inner_dim = input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, input(1).count())
        << "\nnumber of predictions must match the number of labels.";
    this->valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    this->losses.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    this->softmax_op->Run();
    this->prob = ws()->GetTensor("_t_" + anchor() + "_softmax_prob");
    scale = ws()->CreateTensor("_t_" + anchor() + "_focal_scale");
    scale->ReshapeLike(*this->prob);
    
    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SparseSoftmaxFocalLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SparseSoftmaxFocalLoss);
#endif
OPERATOR_SCHEMA(SparseSoftmaxFocalLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SparseSoftmaxFocalLossGradientOp<Context>::RunWithType() {
    auto* label_data = input(1).template data<T, Context>();
    auto* prob_data = this->prob->template mutable_data<T, Context>();
    auto* dXdata = output(0)->template mutable_data<T, Context>();
    auto* valid_data = this->valid.template mutable_data<T, Context>();
    auto* scale_data = scale->template mutable_data<T, Context>();

    kernel::SparseSoftmaxFocalLossGrad<T, Context>(output(0)->count(), 
                                                 output(0)->dim(axis),
                                                            outer_dim, 
                                                            inner_dim, 
                                                                gamma,
                                                                  eps,
                                                           scale_data,
                                                            prob_data,
                                                           label_data,
                                                           valid_data, 
                                                        &this->ignore, 
                                                              dXdata);

    if (normalization == "UNIT") {
        auto* dYdata = input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(input(0).count() / input(0).dim(axis),
                                                       input(0).dim(axis), 
                                                                inner_dim, 
                                                                      1.0, 
                                                                   dYdata, 
                                                               prob_data);
        math::Mul<T, Context>(output(0)->count(), prob_data, dXdata, dXdata);
        return;
    }

    T normalizer;
    if (normalization == "VALID") normalizer = math::ASum<T, Context>(this->valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = outer_dim;
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    auto* dYdata = input(-1).template data<T, CPUContext>();
    math::Scal<T, Context>(output(0)->count(), dYdata[0] / normalizer, dXdata);
}

template <class Context>
void SparseSoftmaxFocalLossGradientOp<Context>::RunOnDevice() {
    this->prob = ws()->GetTensor("_t_" + anchor() + "_softmax_prob");
    scale = ws()->GetTensor("_t_" + anchor() + "_focal_scale");
    outer_dim = this->prob->count(0, axis);
    inner_dim = this->prob->count(axis + 1);
    output(0)->ReshapeLike(input(0));
    this->valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));

    if (input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "unsupported input types.";
}

DEPLOY_CPU(SparseSoftmaxFocalLossGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(SparseSoftmaxFocalLossGradient);
#endif
OPERATOR_SCHEMA(SparseSoftmaxFocalLossGradient).NumInputs(3).NumOutputs(1);

class GetSparseSoftmaxFocalLossGradient final : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetSparseSoftmaxFocalLossGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), GO(0)},
            vector<string> {GI(0)});
    }
};
REGISTER_GRADIENT(SparseSoftmaxFocalLoss, GetSparseSoftmaxFocalLossGradient);

}    // namespace dragon