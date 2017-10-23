#include "operators/update/update_op_base.h"
#include "core/workspace.h"
#include "utils/math_functions.h"

namespace dragon {

template <class Context>
float UpdateOpBase<Context>::param(const string& name) const {
    return ws()->GetTensor(domain + name)
               ->template mutable_data<float, CPUContext>()[0];
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::PreprocessRunWithType() {
    //  scale
    scale_factor = param("scale_gradient");
    if (scale_factor != 1) {
        auto* dXdata = input(0).template mutable_data<T, Context>();
        math::Scal<T, Context>(input(0).count(), scale_factor, dXdata);
    }
    //  clip
    clip_thresh = param("clip_gradient");
    if (clip_thresh > 0) {
        auto* dXdata = input(0).template mutable_data<T, Context>();
        T sumsq_grad = math::Dot<T, Context>(input(0).count(), dXdata, dXdata);
        const T l2norm = sqrt(sumsq_grad);
        if (l2norm > clip_thresh) {
            T factor = clip_thresh / l2norm;
            math::Scal<T, Context>(input(0).count(), factor, dXdata);
        }
    }
    //  decay
    l2_decay = param("l2_decay") * decay_mult;
    if (l2_decay > 0) {
        auto* dXdata = input(0).template mutable_data<T, Context>();
        auto* Xdata = output(0)->template data<T, Context>();
        math::Axpy<T, Context>(input(0).count(), l2_decay, Xdata, dXdata);
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::UpdateRunWithType() {
    auto* dXdata = input(0).template mutable_data<T, Context>();
    auto* Xdata = output(0)->template mutable_data<T, Context>();
    math::Axpy<T, Context>(output(0)->count(), -1.0, dXdata, Xdata);
    math::Set<T, Context>(input(0).count(), 0, dXdata);
}


template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
    CHECK(input(0).dims() == output(0)->dims())
        << "\nTensor and its gradient must have same dims if update.";
    if (input(0).count() == 0 || output(0)->count() == 0) return;
    if (input(0).template IsType<float>()) {
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