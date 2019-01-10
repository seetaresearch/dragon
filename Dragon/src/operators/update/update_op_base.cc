#include "core/workspace.h"
#include "utils/cast.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/update/update_op_base.h"

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
    // Scale
    scale_factor = Param("scale_gradient");
    if (scale_factor != 1.f) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        math::Scale(Input(0).count(),
            scale_factor, dXdata, dXdata, ctx());
    }
    // Clip
    clip_thresh = Param("clip_gradient");
    if (clip_thresh > 0) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        T sumsq_grad;
        math::Dot(Input(0).count(), dXdata, dXdata, &sumsq_grad, ctx());
        const float l2norm = sqrt(cast::to<float>(sumsq_grad));
        if (l2norm > clip_thresh) {
            float norm_factor = clip_thresh / l2norm;
            math::Scale(Input(0).count(),
                norm_factor, dXdata, dXdata, ctx());
        }
    }
    // L2 Decay
    l2_decay = Param("l2_decay") * decay_mult;
    if (l2_decay > 0) {
        auto* dXdata = Input(0).template mutable_data<T, Context>();
        auto* Xdata = Output(0)->template data<T, Context>();
        math::Axpy(Input(0).count(), l2_decay, Xdata, dXdata, ctx());
    }
}

template <class Context>
void UpdateOpBase<Context>::UpdateRunWithFloat32() {
    auto* dXdata = Input(0).template mutable_data<float, Context>();
    auto* Xdata = Output(0)->template mutable_data<float, Context>();
    // Weights Update & Zero Grads
    math::Axpy(Output(0)->count(), -1, dXdata, Xdata, ctx());
    if (zero_grad) math::Set(Input(0).count(), 0.f, dXdata, ctx());
}

template <class Context>
void UpdateOpBase<Context>::UpdateRunWithFloat16() {

    /*!
     * -----------------------------------------------
     *
     *            Mixed Precision Training
     *
     *         http://arxiv.org/abs/1710.03740
     *
     * ------------------------------------------------
     */

    // The "master" updates
    auto* dX32T = ws()->GetTensor(Input(0).name() + "/f32");
    auto* dX32 = dX32T->template data<float, Context>();

    // The "fp16" weights && grads
    auto* X16 = Output(0)->template mutable_data<float16, Context>();
    auto* dX16 = Input(0).template mutable_data<float16, Context>();

    // Apply Update
    kernel::MixedPrecisionUpdate(Output(0)->count(), dX32, X16, dX16, ctx());
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
    // Skip empty param or grads
    if (Input(0).count() == 0 || Output(0)->count() == 0) return;
    CHECK(Input(0).dims() == Output(0)->dims())
        << "\nTensor and its gradients should have same dims.\nGot "
        << Output(0)->DimString() << " and " << Input(0).DimString();
    if (XIsType(Input(0), float)) {
        PreprocessRunWithType<float>();
        ComputeRunWithFloat32();
        UpdateRunWithFloat32();
    } else if (XIsType(Input(0), float16)) {
        PreprocessRunWithType<float16>();
        ComputeRunWithFloat16();
        UpdateRunWithFloat16();
    } else LOG(FATAL) << DTypeHelper(Input(0), { "float32", "float16" });
} 

template class UpdateOpBase<CPUContext>;
#ifdef WITH_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

}  // namespace dragon