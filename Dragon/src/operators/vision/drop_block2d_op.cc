#include "core/workspace.h"
#include "utils/op_kernel.h"
#include "utils/math_functions.h"
#include "operators/vision/drop_block_op.h"

namespace dragon {

template <class Context> template <typename T>
void DropBlock2dOp<Context>::RunImpl() {
    if (phase() == "TEST") {
        Y(0)->CopyFrom(X(0), ctx());
    } else if (phase() == "TRAIN") {
        if (data_format() == "NCHW") {
            n_ = X(0).dim(0), c_ = X(0).dim(1);
            h_ = X(0).dim(2), w_ = X(0).dim(3);
        } else if (data_format() == "NHWC") {
            n_ = X(0).dim(0), c_ = X(0).dim(-1);
            h_ = X(0).dim(1), w_ = X(0).dim(2);
        }

        seed_h_ = h_ - block_size_ + 1;
        seed_w_ = w_ - block_size_ + 1;

        CHECK(seed_h_ > 0 && seed_w_ > 0)
            << "\nExcepted block_size <= feat_size.";

        if (dec_ > 0.f && prob_ > keep_prob()) {
            prob_ -= dec_;
        } else { prob_ = keep_prob(); }

        gamma_= (1.f - prob_) / (block_size_ * block_size_);
        gamma_ *= (alpha_ * (h_ * w_) / (seed_h_ * seed_w_));

        auto count = X(0).count();

        auto* mask = ws()
            ->CreateTensor(unique_name("mask"))
            ->ReshapeLike(X(0))
            ->template mutable_data<uint8_t, Context>();

        auto* norm = ws()
            ->CreateTensor(unique_name("norm"))
            ->Reshape({})
            ->template mutable_data<float, CPUContext>();

        auto buf = ws()->template data<Context>({
            n_ * c_ * seed_h_ * seed_w_ * sizeof(uint32_t),
            count * sizeof(int),
            count * sizeof(float)
        });

        // Fill the mask with ones
        math::Set(count, 1, (int*)buf[1], ctx());

        // Generate 2d mask from seed region
        kernel::DropBlock2d(
            n_, c_, h_, w_,
            seed_h_, seed_w_,
            block_size_, gamma_,
            data_format(),
            (uint32_t*)buf[0],
            (int*)buf[1], ctx()
        );

        // Convert to float mask for counting
        kernel::TypeA2B(
            count,
            (int*)buf[1],
            (float*)buf[2], ctx()
        );

        // Convert to uint8 mask for applying
        kernel::TypeA2B(
            count,
            (int*)buf[1],
            mask, ctx()
        );

        // Count && Apply
        auto normalizer = math::Sum(
            count, 1.f,
            (float*)buf[2], ctx()
        );
        normalizer = std::max(normalizer, 1.f);
        norm[0] = normalizer = count / normalizer;

        auto* x = X(0).template data<T, Context>();
        auto* y = Y(0)->template mutable_data<T, Context>();

        kernel::ApplyMask(
            count,
            normalizer,
            x, mask,
            y, ctx()
        );
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void DropBlock2dOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

template <class Context> template <typename T>
void DropBlock2dGradientOp<Context>::RunImpl() {
    auto* mask = ws()
        ->GetTensor(unique_name("mask"))
        ->template data<uint8_t, Context>();

    auto* norm = ws()
        ->GetTensor(unique_name("norm"))
        ->template data<float, CPUContext>();

    auto* dy = X(1).template data<T, Context>();
    auto* dx = Y(0)->template mutable_data<T, Context>();

    if (phase() == "TEST") {
        NOT_IMPLEMENTED;
    } else if (phase() == "TRAIN") {
        kernel::ApplyMask(
            Y(0)->count(),
            norm[0],
            dy, mask,
            dx, ctx()
        );
    } else {
        LOG(FATAL) << "Unknown Phase: " << phase();
    }
}

template <class Context>
void DropBlock2dGradientOp<Context>::RunOnDevice() {
    Y(0)->ReshapeLike(X(0));

    if (XIsType(X(0), float)) {
        RunImpl<float>();
    } else if (XIsType(X(0), float16)) {
        RunImpl<float16>();
    } else {
        LOG(FATAL) << DTypeString(X(0),
            { "float32", "float16" }
        );
    }
}

DEPLOY_CPU(DropBlock2d);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropBlock2d);
#endif

OPERATOR_SCHEMA(DropBlock2d)
     /* X */
    .NumInputs(1)
     /* Y */
    .NumOutputs(1)
     /* X => Y */
    .Inplace({ { 0, 0 } });

DEPLOY_CPU(DropBlock2dGradient);
#ifdef WITH_CUDA
DEPLOY_CUDA(DropBlock2dGradient);
#endif

OPERATOR_SCHEMA(DropBlock2dGradient)
     /* Y, dY */
    .NumInputs(2)
     /* dX */
    .NumOutputs(1)
     /* dY => dX */
    .Inplace({ { 1, 0 } });

REGISTER_GRADIENT(DropBlock2d, InplaceGradientMaker);

}  // namepsace dragon