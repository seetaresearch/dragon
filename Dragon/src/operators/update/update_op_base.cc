#include "core/workspace.h"
#include "utils/cast.h"
#include "utils/math_functions.h"
#include "utils/op_kernel.h"
#include "operators/update/update_op_base.h"

namespace dragon {

template <class Context>
float UpdateOpBase<Context>::param(const string& name) const {
    return ws()->GetTensor(slot_ + "/" + name)
        ->template mutable_data<float, CPUContext>()[0];
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::Process(Tensor* dX, Tensor* X) {
    // Scale
    auto scale_factor = param("scale_gradient");
    if (scale_factor != 1.f) {
        auto* dx = dX->template mutable_data<T, Context>();
        math::Scale(dX->count(), scale_factor, dx, dx, ctx());
    }
    // Clip
    auto clip_thresh = param("clip_gradient");
    if (clip_thresh > 0.f) {
        T sumsq_grad;
        auto* dx = dX->template mutable_data<T, Context>();
        math::Dot(dX->count(), dx, dx, &sumsq_grad, ctx());
        auto l2_norm = sqrt(cast::to<float>(sumsq_grad));
        if (l2_norm > clip_thresh) {
            math::Scale(
                dX->count(),
                clip_thresh / l2_norm,
                dx, dx, ctx()
            );
        }
    }
    // L2 Decay
    auto l2_decay = param("l2_decay") * decay_mult_;
    if (l2_decay > 0) {
        if (XIsType((*X), float16)) {
            auto* x  = X->template data<float16, Context>();
            auto* dx = dX->template mutable_data<float, Context>();
            kernel::MixedPrecL2Decay(
                X->count(), l2_decay, x, dx, ctx());
        } else {
            auto* x  = X->template data<T, Context>();
            auto* dx = dX->template mutable_data<T, Context>();
            math::Axpy(X->count(), l2_decay, x, dx, ctx());
        }
    }
}

template <class Context> template <typename T>
void UpdateOpBase<Context>::Apply(Tensor* dX, Tensor* X) {
    if (XIsType((*X), float16)) {
        auto* dx = dX->template data<float, Context>();
        auto* x  = X->template mutable_data<float16, Context>();
        kernel::MixedPrecUpdate(X->count(), dx, x, ctx());
    } else {
        auto* dx = dX->template data<T, Context>();
        auto* x  = X->template mutable_data<T, Context>();
        math::Axpy(X->count(), -1.f, dx, x, ctx());
    }
}

template <class Context>
void UpdateOpBase<Context>::RunOnDevice() {
    // Skip empty param or grad
    if (X(0).count() == 0 || Y(0)->count() == 0) return;

    CHECK(X(0).dims() == Y(0)->dims())
        << "\nParam and Grad should have same dimensions."
        << "\nGot" << Y(0)->DimString()
        << " and " << X(0).DimString();

    if (XIsType(X(0), float)) {
        Process<float>(&X(0), Y(0));
        Compute(&X(0));
        Apply<float>(&X(0), Y(0));
    } else if (XIsType(X(0), float16)) {
        auto* MdX = ws()
            ->CreateTensor(X(0).name() + "/master")
            ->ReshapeLike(X(0));
        kernel::TypeA2B(
            X(0).count(),
            X(0).template data<float16, Context>(),
            MdX->template mutable_data<float, Context>(),
            ctx()
        );
        Process<float>(MdX, Y(0));
        Compute(MdX);
        Apply<float>(MdX, Y(0));
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template class UpdateOpBase<CPUContext>;
#ifdef WITH_CUDA
template class UpdateOpBase<CUDAContext>;
#endif

}  // namespace dragon