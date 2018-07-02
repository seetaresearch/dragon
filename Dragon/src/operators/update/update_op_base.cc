#include "operators/update/update_op_base.h"
#include "core/workspace.h"
#include "utils/math_functions.h"
#include "utils/cast.h"

namespace dragon {

template <class Context>
float UpdateOpBase<Context>::Param(const string& name) const {
    return ws()->GetTensor(slot + "/" + name)
        ->template mutable_data<float, CPUContext>()[0];
}

template <class Context>
string UpdateOpBase<Context>::Slot() {
    return slot + "/" + Output(0)->name();
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::PreprocessRunWithType() {
    //  scale
    scale_factor = Param("scale_gradient");
    if (scale_factor != 1) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        math::Scal<T, Context>(Input(0).count(),
            scale_factor, dXdata, &ctx());
    }
    //  clip
    clip_thresh = Param("clip_gradient");
    if (clip_thresh > 0) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        float sumsq_grad = math::Dot<T, Context>(
            Input(0).count(), dXdata, dXdata, &ctx());
        const float l2norm = sqrt(sumsq_grad);
        if (l2norm > clip_thresh) {
            float norm_factor = clip_thresh / l2norm;
            math::Scal<T, Context>(Input(0).count(),
                norm_factor, dXdata, &ctx());
        }
    }
    //  decay
    l2_decay = Param("l2_decay") * decay_mult;
    if (l2_decay > 0) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        auto* Xdata = Output(0)->template data<T, Context>();
        math::Axpy<T, Context>(Input(0).count(),
            l2_decay, Xdata, dXdata, &ctx());
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::UpdateRunWithType() {
    auto* dXdata = Input(0).template mutable_data<T, Context>();
    auto* Xdata = Output(0)->template mutable_data<T, Context>();
    math::Axpy<T, Context>(Output(0)->count(), -1, dXdata, Xdata, &ctx());
    T zeroT = dragon_cast<T, float>(0.f);
    if (zero_grad) math::Set<T, Context>(Input(0).count(), zeroT, dXdata);
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
    //  skip empty param or grad
    if (Input(0).count() == 0 || Output(0)->count() == 0) return;
    CHECK(Input(0).dims() == Output(0)->dims())
        << "\nTensor and its gradients should have same dims.\nGot "
        << Output(0)->DimString() << " and " << Input(0).DimString();
    if (XIsType(Input(0), float)) {
        PreprocessRunWithType<float>();
        ComputeRunWithFloat();
        UpdateRunWithType<float>();
    } else if (XIsType(Input(0), float16)) {
        PreprocessRunWithType<float16>();
        ComputeRunWithFloat16();
        UpdateRunWithType<float16>();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
} 

template class UpdateOpBase<CPUContext>;
#ifdef WITH_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

}    // namespace dragon