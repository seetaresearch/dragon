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
    auto* label_data = Input(1).template data<T, Context>();
    auto* loss_data = this->losses.template mutable_data<T, Context>();
    auto* valid_data = this->valid.template mutable_data<T, Context>();
    auto* scale_data = scale->template mutable_data<T, Context>();

    kernel::SparseSoftmaxFocalLoss<T, Context>(Input(0).count(), 
                                             Input(0).dim(axis),
                                                      outer_dim, 
                                                      inner_dim,
                                                      pos_alpha,
                                                      neg_alpha,
                                                          gamma,
                                                         neg_id,
                                                      prob_data, 
                                                     label_data, 
                                                     scale_data,
                                                      loss_data, 
                                                     valid_data, 
                                                 &this->ignore);

    if (normalization == "UNIT") {
        Output(0)->ReshapeLike(this->losses);
        Output(0)->Share(this->losses);
        return;
    }

    T normalizer;
    if (normalization == "VALID")
        normalizer = math::ASum<T, Context>(this->valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    T loss = math::ASum<T, Context>(this->losses.count(), loss_data);
    Output(0)->Reshape(vector<TIndex>(1, 1));
    auto* Ydata = Output(0)->template mutable_data<T, CPUContext>();
    Ydata[0] = loss / normalizer;
}

template <class Context>
void SparseSoftmaxFocalLossOp<Context>::RunOnDevice() {
    outer_dim = Input(0).count(0, axis);
    inner_dim = Input(0).count(axis + 1);
    CHECK_EQ(outer_dim * inner_dim, Input(1).count())
        << "\nNumber of predictions must match the number of labels.";
    this->valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    this->losses.Reshape(vector<TIndex>(1, outer_dim * inner_dim));
    this->softmax_op->Run();
    this->prob = ws()->GetTensor("/mnt/" + Anchor() + "/softmax/prob");
    scale = ws()->CreateTensor("/mnt/" + Anchor() + "/focal/scale");
    scale->ReshapeLike(*this->prob);
    
    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
}

DEPLOY_CPU(SparseSoftmaxFocalLoss);
#ifdef WITH_CUDA
DEPLOY_CUDA(SparseSoftmaxFocalLoss);
#endif
OPERATOR_SCHEMA(SparseSoftmaxFocalLoss).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void SparseSoftmaxFocalLossGradientOp<Context>::RunWithType() {
    auto* label_data = Input(1).template data<T, Context>();
    auto* prob_data = this->prob->template mutable_data<T, Context>();
    auto* dXdata = Output(0)->template mutable_data<T, Context>();
    auto* valid_data = this->valid.template mutable_data<T, Context>();
    auto* scale_data = scale->template mutable_data<T, Context>();

    kernel::SparseSoftmaxFocalLossGrad<T, Context>(Output(0)->count(), 
                                                 Output(0)->dim(axis),
                                                            outer_dim, 
                                                            inner_dim, 
                                                                gamma,
                                                               neg_id,
                                                                  eps,
                                                           scale_data,
                                                            prob_data,
                                                           label_data,
                                                           valid_data, 
                                                        &this->ignore, 
                                                              dXdata);

    if (normalization == "UNIT") {
        auto* dYdata = Input(-1).template data<T, Context>();
        kernel::SumGrad<T, Context>(Input(0).count() / Input(0).dim(axis),
                                                       Input(0).dim(axis), 
                                                                inner_dim, 
                                                                      1.0, 
                                                                   dYdata, 
                                                               prob_data);
        math::Mul<T, Context>(Output(0)->count(), prob_data, dXdata, dXdata);
        return;
    }

    T normalizer;
    if (normalization == "VALID") normalizer = math::ASum<T, Context>(this->valid.count(), valid_data);
    else if (normalization == "BATCH_SIZE") normalizer = Input(0).dim(0);
    else if (normalization == "FULL") normalizer = outer_dim * inner_dim;
    else if (normalization == "NONE") normalizer = 1;
    auto* dYdata = Input(-1).template data<T, CPUContext>();
    math::Scal<T, Context>(Output(0)->count(), dYdata[0] / normalizer, dXdata);
}

template <class Context>
void SparseSoftmaxFocalLossGradientOp<Context>::RunOnDevice() {
    this->prob = ws()->GetTensor("/mnt/" + Anchor() + "/softmax/prob");
    scale = ws()->GetTensor("/mnt/" + Anchor() + "/focal/scale");
    outer_dim = this->prob->count(0, axis);
    inner_dim = this->prob->count(axis + 1);
    Output(0)->ReshapeLike(Input(0));
    this->valid.Reshape(vector<TIndex>(1, outer_dim * inner_dim));

    if (Input(0).template IsType<float>()) RunWithType<float>();
    else LOG(FATAL) << "Unsupported input types.";
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