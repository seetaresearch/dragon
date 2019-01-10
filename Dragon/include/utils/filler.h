/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *      <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_FILLER_H_
#define DRAGON_UTILS_FILLER_H_

#include "core/registry.h"
#include "utils/math_functions.h"

namespace dragon {

template <typename T, class Context>
class Filler {
 public:
    Filler(const TensorFillerProto& filler_proto)
        : filler_proto_(filler_proto) {}

    virtual void Fill(Tensor* tensor, Context* ctx) = 0;

    inline TensorFillerProto& filler_proto() { return filler_proto_; }

 protected:
    TensorFillerProto filler_proto_;
};

template <typename T, class Context>
class ConstantFiller final : public Filler<T, Context> {
 public:
    ConstantFiller(const TensorFillerProto& filler_proto)
        : Filler<T, Context>(filler_proto) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        math::Set<T, Context>(tensor->count(),
            cast::to<T>(filler_proto().value()),
                tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler_proto;
};

template <typename T, class Context>
class NormalFiller final : public Filler<T, Context> {
 public:
    NormalFiller(const TensorFillerProto& filler_proto)
        : Filler<T, Context>(filler_proto) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        math::RandomNormal<T, Context>(tensor->count(),
            filler_proto().mean(), filler_proto().std(),
                tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler_proto;
};

template <typename T, class Context>
class TruncatedNormalFiller final : public Filler<T, Context> {
 public:
    TruncatedNormalFiller(const TensorFillerProto& filler_proto)
        : Filler<T, Context>(filler_proto) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        // It's difficult to implement it on gpu
        static CPUContext cctx;
        math::RandomTruncatedNormal<T, CPUContext>(tensor->count(),
            filler_proto().mean(), filler_proto().std(),
                filler_proto().low(), filler_proto().high(),
                    tensor->mutable_data<T, CPUContext>(), &cctx);
    }

 protected:
    using Filler<T, Context>::filler_proto;
};

template <typename T, class Context>
class UniformFiller final : public Filler<T, Context> {
 public:
    UniformFiller(const TensorFillerProto& filler_proto)
        : Filler<T, Context>(filler_proto) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        math::RandomUniform<T, Context>(tensor->count(),
            filler_proto().low(), filler_proto().high(),
                tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler_proto;
};

template <typename T, class Context>
class XavierFiller final : public Filler<T, Context> {
 public:
    XavierFiller(const TensorFillerProto& filler_proto)
        : Filler<T, Context>(filler_proto) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        auto fan_in = tensor->count() / tensor->dim(0);
        auto fan_out = tensor->count() / tensor->dim(1);
        float n = (float)fan_in, scale = 3.f;
        if (filler_proto().has_scale()) scale = filler_proto().scale();
        if (filler_proto().variance_norm() ==
            TensorFillerProto_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / 2.f;
        } else if (filler_proto().variance_norm() ==
            TensorFillerProto_VarianceNorm_FAN_OUT) {
            n = (float)fan_out;
        }

        float limit = std::sqrt(scale / n);
        math::RandomUniform<T, Context>(tensor->count(),
            -limit, limit, tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler_proto;
};

template <typename T, class Context>
class MSRAFiller final : public Filler <T, Context> {
 public:
    MSRAFiller(const TensorFillerProto& filler_proto)
        : Filler<T, Context>(filler_proto) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        auto fan_in = tensor->count() / tensor->dim(0);
        auto fan_out = tensor->count() / tensor->dim(1);
        float n = (float)fan_in, scale = 2.f;
        if (filler_proto().has_scale()) scale = filler_proto().scale();
        if (filler_proto().variance_norm() ==
            TensorFillerProto_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / 2.f;
        } else if (filler_proto().variance_norm() == 
            TensorFillerProto_VarianceNorm_FAN_OUT) {
            n = (float)fan_out;
        }
        float std = std::sqrt(scale / n);
        math::RandomNormal<T, Context>(tensor->count(),
            0.f, std, tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler_proto;
};

template <typename T, class Context>
Filler<T, Context>* CreateFiller(const TensorFillerProto& filler_proto) {
    const string& type = filler_proto.type();
    if (type == "constant") {
        return new ConstantFiller<T, Context>(filler_proto);
    } else if (type == "uniform") {
        return new UniformFiller<T, Context>(filler_proto);
    } else if (type == "normal") {
        return new NormalFiller<T, Context>(filler_proto);
    } else if (type == "truncated_normal") {
        return new TruncatedNormalFiller<T, Context>(filler_proto);
    } else if (type == "xavier" || type == "glorot_uniform") {
        return new XavierFiller<T, Context>(filler_proto);
    } else if (type == "msra" || type == "glorot_normal") {
        return new MSRAFiller<T, Context>(filler_proto);
    } 
    return new ConstantFiller<T, Context>(filler_proto);
}

}  // namespace dragon

#endif  // DRAGON_UTILS_FILLER_H_