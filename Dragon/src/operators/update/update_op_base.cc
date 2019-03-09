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

template <class Context> template <typename T>
void UpdateOpBase<Context>::ProcessGradients(Tensor* dX, Tensor* X) {
    // Scale
    scale_factor = Param("scale_gradient");
    if (scale_factor != 1.f) {
        auto* dXdata = dX->template mutable_data<T, Context>();
        math::Scale(dX->count(), scale_factor, dXdata, dXdata, ctx());
    }
    // Clip
    clip_thresh = Param("clip_gradient");
    if (clip_thresh > 0) {
        auto* dXdata = dX->template mutable_data<T, Context>();
        T sumsq_grad;
        math::Dot(dX->count(), dXdata, dXdata, &sumsq_grad, ctx());
        auto l2norm = sqrt(cast::to<float>(sumsq_grad));
        if (l2norm > clip_thresh) {
            math::Scale(dX->count(),
                clip_thresh / l2norm,
                    dXdata, dXdata, ctx());
        }
    }
    // L2 Decay
    l2_decay = Param("l2_decay") * decay_mult;
    if (l2_decay > 0) {
        if (XIsType((*X), float16)) {
            auto* dXdata = dX->template mutable_data<float, Context>();
            auto* Xdata = X->template data<float16, Context>();
            kernel::MixedPrecisionL2Decay(
                X->count(), l2_decay, Xdata, dXdata, ctx());
        } else {
            auto* dXdata = dX->template mutable_data<T, Context>();
            auto* Xdata = X->template data<T, Context>();
            math::Axpy(X->count(), l2_decay, Xdata, dXdata, ctx());
        }
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::ApplyUpdates(Tensor* dX, Tensor* X) {
    if (XIsType((*X), float16)) {
        auto* dXdata = dX->template data<float, Context>();
        auto* Xdata = X->template mutable_data<float16, Context>();
        kernel::MixedPrecisionUpdate(
            X->count(), dXdata, Xdata, ctx());
    } else {
        auto* dXdata = dX->template data<T, Context>();
        auto* Xdata = X->template mutable_data<T, Context>();
        math::Axpy(X->count(), -1.f, dXdata, Xdata, ctx());
    }
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
    // Skip empty param or grads
    if (Input(0).count() == 0 || Output(0)->count() == 0) return;

    CHECK(Input(0).dims() == Output(0)->dims())
        << "\nTensor and its gradients should have same dims.\nGot "
        << Output(0)->DimString() << " and " << Input(0).DimString();

    if (XIsType(Input(0), float)) {
        ProcessGradients<float>(&Input(0), Output(0));
        ComputeUpdates(&Input(0));
        ApplyUpdates<float>(&Input(0), Output(0));
    } else if (XIsType(Input(0), float16)) {
        Tensor* MasterDX = ws()->CreateTensor(
            Input(0).name() + "/f32")
                ->ReshapeLike(Input(0));
        auto* dXdata = Input(0).template data<float16, Context>();
        auto* MdXdata = MasterDX->template mutable_data<float, Context>();
        kernel::TypeA2B(MasterDX->count(), dXdata, MdXdata, ctx());
        ProcessGradients<float>(MasterDX, Output(0));
        ComputeUpdates(MasterDX);
        ApplyUpdates<float>(MasterDX, Output(0));
    } else {
        LOG(FATAL) << DTypeHelper(Input(0),
            { "float32", "float16" });
    }
}

template class UpdateOpBase<CPUContext>;
#ifdef WITH_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

}  // namespace dragon