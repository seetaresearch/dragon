#include "operators/update/update_op_base.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
float UpdateOpBase<Context>::Param(const string& name) const {
    return ws()->GetTensor(domain + name)
               ->template mutable_data<float, CPUContext>()[0];
}

template <class Context>
string UpdateOpBase<Context>::Slot() {
    const string slot = OperatorBase::GetSingleArg<string>("slot", "");
    return slot.empty() ? name() : slot;
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::PreprocessRunWithType() {
    //  scale
    scale_factor = Param("scale_gradient");
    if (scale_factor != 1) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        math::Scal<T, Context>(Input(0).count(), scale_factor, dXdata);
    }
    //  clip
    clip_thresh = Param("clip_gradient");
    if (clip_thresh > 0) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        T sumsq_grad = math::Dot<T, Context>(Input(0).count(), dXdata, dXdata);
        const T l2norm = sqrt(sumsq_grad);
        if (l2norm > clip_thresh) {
            T factor = clip_thresh / l2norm;
            math::Scal<T, Context>(Input(0).count(), factor, dXdata);
        }
    }
    //  decay
    l2_decay = Param("l2_decay") * decay_mult;
    if (l2_decay > 0) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        auto* Xdata = Output(0)->template data<T, Context>();
        math::Axpy<T, Context>(Input(0).count(), l2_decay, Xdata, dXdata);
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::UpdateRunWithType() {
    auto* dXdata = Input(0).template mutable_data<T, Context>();
    auto* Xdata = Output(0)->template mutable_data<T, Context>();
    math::Axpy<T, Context>(Output(0)->count(), -1.0, dXdata, Xdata);
    math::Set<T, Context>(Input(0).count(), 0, dXdata);
}


template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
    CHECK(Input(0).dims() == Output(0)->dims())
        << "\nTensor and its gradient must have same dims if update.";
    if (Input(0).count() == 0 || Output(0)->count() == 0) return;
    if (Input(0).template IsType<float>()) {
        PreprocessRunWithType<float>();
        ComputeRunWithFloat();
        UpdateRunWithType<float>();
    } else {
        LOG(FATAL) << "Unsupported input types.";
    }
} 

template class UpdateOpBase<CPUContext>;
#ifdef WITH_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

}    // namespace dragon