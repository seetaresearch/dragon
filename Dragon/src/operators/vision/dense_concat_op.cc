#include "operators/vision/dense_concat_op.h"
#include "core/workspace.h"
#include "utils/op_kernel.h"

namespace dragon {

DEPLOY_CPU(DenseConcat);
#ifdef WITH_CUDA
DEPLOY_CUDA(DenseConcat);
#endif
OPERATOR_SCHEMA(DenseConcat).NumInputs(2).NumOutputs(1);

template <class Context> template <typename T>
void DenseConcatGradientOp<Context>::RestoreX1() {
    CHECK_GT(growth_rate, 0) << "\nInvalid growth rate, please preset it.";
    this->concat_dims = input(-1).dims();
    this->y_concat_dim = this->concat_dims[this->axis];
    this->outer_dim = input(-1).count(0, this->axis);
    this->inner_dim = input(-1).count(this->axis + 1);
    this->concat_dims[this->axis] -= growth_rate;
    input(0).Reshape(this->concat_dims);
    this->x_concat_dim = input(0).dim(this->axis);
    TIndex count = input(0).count();
    auto* Ydata = input(-2).template data<T, Context>();
    auto* Xdata = input(0).template mutable_data<T, Context>();
    kernel::ConcatGrad<T, Context>(count,
                         this->outer_dim, 
                         this->inner_dim,
                      this->x_concat_dim, 
                      this->y_concat_dim,
                                       0,
                                   Ydata, 
                                   Xdata,
                                 &ctx());
}

template <class Context>
void DenseConcatGradientOp<Context>::ElimateCorruption() {
    Set<string> all_heads;
    queue<int> safe_heads;
    Tensor* head = ws()->GetTensor("_t_mirror_stage_head");
    string* head_data = head->mutable_data<string, CPUContext>();
    for (int i = 0; i < head->count(); i++) all_heads.insert(head_data[i]);

    //  sub-graph run
    if (input(0).is_corrupted() && !all_heads.count(input(0).name())) {
        //  pre-process
        LOG(DEBUG) << "Tensor(" << input(0).name() << ") is corrupted, recompute...  ";
        for (int i = 0; i < head->count(); i++) {
            bool safe = true;
            for (int j = 0; j < InputSize(); j++)
                if (head_data[i] == input(j).name()) safe = false;
            if (safe) safe_heads.push(i);
        }
        int idx = safe_heads.front();
        safe_heads.pop();
        Tensor* buffer = ws()->GetTensor("_t_mirror_stage_buffer_" + dragon_cast<string, int>(idx));
        input(0).Move(buffer->memory());
        head_data[idx] = input(0).name();
        if (input(-2).template IsType<float>()) RestoreX1<float>();
#ifdef WITH_CUDA_FP16
        else if (input(-2).template IsType<float16>()) RestoreX1<float16>();
#endif
        else LOG(FATAL) << "Unsupported input types.";
        //  post-process
        if (input(0).memory() != buffer->memory()) buffer->Move(input(0).memory());
    }

    //  check available head
    while (!safe_heads.empty()) safe_heads.pop();
    all_heads.clear();
    for (int i = 0; i < head->count(); i++) {
        bool safe = true;
        for (int j = 0; j < InputSize(); j++) 
            if (head_data[i] == input(j).name()) safe = false;
        if (safe) safe_heads.push(i);
        all_heads.insert(head_data[i]);
    }

    //  pre-process
    for (int i = 0; i < OutputSize(); i++) {
        if (output(i)->is_corrupted()) {
            bool inplace_flag = false;
            for (int j = 0; j < InputSize(); j++)
                if (output(i)->name() == input(j).name()) inplace_flag = true;
            if (inplace_flag || all_heads.count(output(i)->name())) continue;    //  skip to use new buffer
            CHECK(!safe_heads.empty())
                << "\nAt most (" << safe_heads.size() << " [safe] / "
                << all_heads.size() << " [total] can be used for corrupted output in "
                << "(" << name() << ", " << type() << "), "
                << "\nadd WORKSPACE_MAX_CORRUPTED_SIZE for more powerful mirror stage ?";
            int idx = safe_heads.front();
            safe_heads.pop();
            Tensor* buffer = ws()->GetTensor("_t_mirror_stage_buffer_" + dragon_cast<string, int>(idx));
            output(i)->Move(buffer->memory());
            head_data[idx] = output(i)->name();
        }
    }
}

DEPLOY_CPU(DenseConcatGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DenseConcatGradient);
#endif
OPERATOR_SCHEMA(DenseConcatGradient).NumInputs(4).NumOutputs(2);

class GetDenseConcatGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(GetDenseConcatGradient);
    vector<OperatorDef> MakeDefs() override {
        return SingleDef(def.type() + "Gradient", "",
            vector<string> {I(0), I(1), O(0), GO(0)},
            vector<string> {GI(0), GI(1)});
    }
};
REGISTER_GRADIENT(DenseConcat, GetDenseConcatGradient);

}   // namespace dragon