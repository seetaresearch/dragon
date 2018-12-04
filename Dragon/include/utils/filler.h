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

#include "protos/dragon.pb.h"
#include "core/registry.h"
#include "utils/math_functions.h"

namespace dragon {

template <typename T, class Context>
class Filler {
 public:
    Filler(const TensorFiller& filler) : filler_(filler) {}

    virtual void Fill(Tensor* tensor, Context* ctx) = 0;

    inline TensorFiller& filler() { return filler_; }

 protected:
    TensorFiller filler_;
};

template <typename T, class Context>
class ConstantFiller final : public Filler<T, Context> {
 public:
    ConstantFiller(const TensorFiller& filler)
        : Filler<T, Context>(filler) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        math::Set<T, Context>(tensor->count(),
            dragon_cast<T, float>(filler().value()),
                tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler;
};

template <typename T, class Context>
class NormalFiller final : public Filler<T, Context> {
 public:
    NormalFiller(const TensorFiller& filler)
        : Filler<T, Context>(filler) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        math::RandomNormal<T, Context>(tensor->count(),
            filler().mean(), filler().std(),
                tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler;
};

template <typename T, class Context>
class TruncatedNormalFiller final : public Filler<T, Context> {
 public:
    TruncatedNormalFiller(const TensorFiller& filler)
        : Filler<T, Context>(filler) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        // It's difficult to implement it on gpu
        static CPUContext cctx;
        math::RandomTruncatedNormal<T, CPUContext>(tensor->count(),
            filler().mean(), filler().std(),
                filler().low(), filler().high(),
                    tensor->mutable_data<T, CPUContext>(), &cctx);
    }

 protected:
    using Filler<T, Context>::filler;
};

template <typename T, class Context>
class UniformFiller final : public Filler<T, Context> {
 public:
    UniformFiller(const TensorFiller& filler) 
        : Filler<T, Context>(filler) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        math::RandomUniform<T, Context>(tensor->count(),
            filler().low(), filler().high(),
                tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler;
};

template <typename T, class Context>
class XavierFiller final : public Filler<T, Context> {
 public:
    XavierFiller(const TensorFiller& filler)
        : Filler<T, Context>(filler) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        auto fan_in = tensor->count() / tensor->dim(0);
        auto fan_out = tensor->count() / tensor->dim(1);
        float n = (float)fan_in, scale = 3.f;
        if (filler().has_scale()) scale = filler().scale();
        if (filler().variance_norm() ==
            TensorFiller_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / 2.f;
        } else if (filler().variance_norm() ==
            TensorFiller_VarianceNorm_FAN_OUT) {
            n = (float)fan_out;
        }
        float limit = std::sqrt(scale / n);
        math::RandomUniform<T, Context>(tensor->count(),
            -limit, limit, tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler;
};

template <typename T, class Context>
class MSRAFiller final : public Filler <T, Context> {
 public:
    MSRAFiller(const TensorFiller& filler)
        : Filler<T, Context>(filler) {}

    void Fill(Tensor* tensor, Context* ctx) override {
        auto fan_in = tensor->count() / tensor->dim(0);
        auto fan_out = tensor->count() / tensor->dim(1);
        float n = (float)fan_in, scale = 2.f;
        if (filler().has_scale()) scale = filler().scale();
        if (filler().variance_norm() ==
            TensorFiller_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / 2.f;
        } else if (filler().variance_norm() == 
            TensorFiller_VarianceNorm_FAN_OUT) {
            n = (float)fan_out;
        }
        float std = std::sqrt(scale / n);
        math::RandomNormal<T, Context>(tensor->count(),
            0.f, std, tensor->mutable_data<T, Context>(), ctx);
    }

 protected:
    using Filler<T, Context>::filler;
};


template <typename T, class Context>
Filler<T, Context>* CreateFiller(const TensorFiller& filler) {
    const string& type = filler.type();
    if (type == "constant") {
        return new ConstantFiller<T, Context>(filler);
    } else if (type == "uniform") {
        return new UniformFiller<T, Context>(filler);
    } else if (type == "normal") {
        return new NormalFiller<T, Context>(filler);
    } else if (type == "truncated_normal") {
        return new TruncatedNormalFiller<T, Context>(filler);
    } else if (type == "xavier") {
        return new XavierFiller<T, Context>(filler);
    } else if (type == "msra") {
        return new MSRAFiller<T, Context>(filler);
    }
    return new ConstantFiller<T, Context>(filler);
}

}  // namespace dragon

#endif  // DRAGON_UTILS_FILLER_H_