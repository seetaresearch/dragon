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
    Filler(const TensorFillerProto& proto)
        : proto_(proto) {}

    virtual void Fill(Tensor* X, Context* ctx) = 0;

    inline TensorFillerProto& proto() { return proto_; }

 protected:
    TensorFillerProto proto_;
};

template <typename T, class Context>
class ConstantFiller final : public Filler<T, Context> {
 public:
    ConstantFiller(const TensorFillerProto& proto)
        : Filler<T, Context>(proto) {}

    void Fill(Tensor* X, Context* ctx) override {
        math::Set(
            X->count(),
            cast::to<T>(proto().value()),
            X->mutable_data<T, Context>(), ctx
        );
    }

 protected:
    using Filler<T, Context>::proto;
};

template <typename T, class Context>
class NormalFiller final : public Filler<T, Context> {
 public:
    NormalFiller(const TensorFillerProto& proto)
        : Filler<T, Context>(proto) {}

    void Fill(Tensor* X, Context* ctx) override {
        math::RandomNormal(
            X->count(),
            proto().mean(), proto().std(),
            X->mutable_data<T, Context>(), ctx
        );
    }

 protected:
    using Filler<T, Context>::proto;
};

template <typename T, class Context>
class TruncatedNormalFiller final : public Filler<T, Context> {
 public:
    TruncatedNormalFiller(const TensorFillerProto& proto)
        : Filler<T, Context>(proto) {}

    void Fill(Tensor* X, Context* ctx) override {
        // It's difficult to implement it on gpu
        math::RandomTruncatedNormal(
            X->count(),
            proto().mean(), proto().std(),
            proto().low(), proto().high(),
            X->mutable_data<T, CPUContext>(), &cctx_
        );
    }

 protected:
    CPUContext cctx_;
    using Filler<T, Context>::proto;
};

template <typename T, class Context>
class UniformFiller final : public Filler<T, Context> {
 public:
    UniformFiller(const TensorFillerProto& proto)
        : Filler<T, Context>(proto) {}

    void Fill(Tensor* X, Context* ctx) override {
        math::RandomUniform(
            X->count(),
            proto().low(), proto().high(),
            X->mutable_data<T, Context>(), ctx
        );
    }

 protected:
    using Filler<T, Context>::proto;
};

template <typename T, class Context>
class XavierFiller final : public Filler<T, Context> {
 public:
    XavierFiller(const TensorFillerProto& proto)
        : Filler<T, Context>(proto) {}

    void Fill(Tensor* X, Context* ctx) override {
        auto fan_in = X->count() / X->dim(0);
        auto fan_out = X->count() / X->dim(1);
        float n = (float)fan_in, scale = 3.f;
        if (proto().has_scale()) scale = proto().scale();
        if (proto().variance_norm() ==
            TensorFillerProto_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / 2.f;
        } else if (proto().variance_norm() ==
            TensorFillerProto_VarianceNorm_FAN_OUT) {
            n = (float)fan_out;
        }
        float limit = std::sqrt(scale / n);
        math::RandomUniform(
            X->count(),
            -limit, limit,
            X->mutable_data<T, Context>(), ctx
        );
    }

 protected:
    using Filler<T, Context>::proto;
};

template <typename T, class Context>
class MSRAFiller final : public Filler <T, Context> {
 public:
    MSRAFiller(const TensorFillerProto& proto)
        : Filler<T, Context>(proto) {}

    void Fill(Tensor* X, Context* ctx) override {
        auto fan_in = X->count() / X->dim(0);
        auto fan_out = X->count() / X->dim(1);
        float n = (float)fan_in, scale = 2.f;
        if (proto().has_scale()) scale = proto().scale();
        if (proto().variance_norm() ==
            TensorFillerProto_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / 2.f;
        } else if (proto().variance_norm() == 
            TensorFillerProto_VarianceNorm_FAN_OUT) {
            n = (float)fan_out;
        }
        float std = std::sqrt(scale / n);
        math::RandomNormal(
            X->count(),
            0.f, std,
            X->mutable_data<T, Context>(), ctx
        );
    }

 protected:
    using Filler<T, Context>::proto;
};

template <typename T, class Context>
Filler<T, Context>* CreateFiller(const TensorFillerProto& proto) {
    const string& type = proto.type();
    if (type == "constant") {
        return new ConstantFiller<T, Context>(proto);
    } else if (type == "uniform") {
        return new UniformFiller<T, Context>(proto);
    } else if (type == "normal") {
        return new NormalFiller<T, Context>(proto);
    } else if (type == "truncated_normal") {
        return new TruncatedNormalFiller<T, Context>(proto);
    } else if (type == "xavier" || type == "glorot_uniform") {
        return new XavierFiller<T, Context>(proto);
    } else if (type == "msra" || type == "glorot_normal") {
        return new MSRAFiller<T, Context>(proto);
    } return new ConstantFiller<T, Context>(proto);
}

}  // namespace dragon

#endif  // DRAGON_UTILS_FILLER_H_