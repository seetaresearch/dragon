#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/array/non_zero_op.h"

#define TENSOR_FROM_VEC(tensor, vec, T) \
    { \
        tensor.Reshape({ (int64_t)vec.size() }); \
        auto* data = tensor.template mutable_data<T, CPUContext>(); \
        for (int i = 0; i < vec.size(); i++) data[i] = (T)vec[i]; \
    }

namespace dragon {

template <class Context> template <typename T>
void NonZeroOp<Context>::RunImpl() {
    auto ndim = X(0).ndim();
    auto nelements = X(0).count();
    auto* x = X(0).template data<T, Context>();

    auto* scratch = ws()->CreateTensor("/share/data");
    auto* indices = ws()->CreateTensor("/share/buffer/grad:0");

    auto* mask = ws()
        ->CreateTensor("/share/buffer/grad:1")
        ->Reshape({ nelements })
        ->template mutable_data<uint8_t, Context>();

    kernel::NotZero(nelements, x, (bool*)mask, ctx());

    kernel::MaskedSelect(
        nelements,
        mask, (T*)nullptr,
        indices, scratch,
        (Tensor*)nullptr, ctx()
    );

    nelements = indices->count();

    auto* y = Y(0)
        ->Reshape({ nelements, (int64_t)ndim })
        ->template mutable_data<int64_t, Context>();

    kernel::UnravelIndex(
        nelements, ndim,
        X_dims_.template data<int, Context>(),
        indices->template data<int64_t, Context>(),
        y, ctx()
    );
}

template <class Context>
void NonZeroOp<Context>::RunOnDevice() {
    TENSOR_FROM_VEC(X_dims_, X(0).dims(), int);

    DispatchHelper<TensorTypes
        <bool, int8_t, uint8_t, int, int64_t,
               float16, float, double>
    >::Call(this, X(0));
}

DEPLOY_CPU(NonZero);
#ifdef WITH_CUDA
DEPLOY_CUDA(NonZero);
#endif

OPERATOR_SCHEMA(NonZero)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1);

NO_GRADIENT(NonZero);

}  // namespace dragon