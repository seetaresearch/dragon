// --------------------------------------------------------
// Dragon
// Copyright(c) 2017 SeetaTech
// Written by Ting Pan
// --------------------------------------------------------

#ifndef DRAGON_UTILS_FILLER_H_
#define DRAGON_UTILS_FILLER_H_

#include "protos/dragon.pb.h"
#include "core/registry.h"
#include "utils/math_functions.h"

namespace dragon {

template <typename T, class Context>
class Filler {
 public:
    Filler(const TensorFiller& filler): filler_(filler) {}
    virtual void Fill(Tensor* tensor) = 0;

    inline TensorFiller& filler() { return filler_; }

 protected:
    TensorFiller filler_;
};


template <typename T, class Context>
class ConstantFiller final : public Filler<T, Context> {
 public:
    ConstantFiller(const TensorFiller& filler): Filler<T, Context>(filler) {}
    void Fill(Tensor* tensor) override {
        math::Set<T, Context>(tensor->count(), 
                              dragon_cast<T, float>(filler().value()), 
                              tensor->mutable_data<T, Context>());
    }
};

template <typename T, class Context>
class NormalFiller final : public Filler<T, Context> {
 public:
    NormalFiller(const TensorFiller& filler): Filler<T, Context>(filler) {}
    void Fill(Tensor* tensor) override {
        math::RandomNormal<T, Context>(tensor->count(), 
                                       filler().mean(),
                                        filler().std(), 
                   tensor->mutable_data<T, Context>());
    }
};

template <typename T, class Context>
class TruncatedNormalFiller final : public Filler < T, Context > {
 public:
    TruncatedNormalFiller(const TensorFiller& filler): Filler<T, Context>(filler) {}
    void Fill(Tensor* tensor) override {
        //  implement of gpu is diffcult 
        math::RandomTruncatedNormal<T, CPUContext>(tensor->count(), 
                                                   filler().mean(), 
                                                    filler().std(), 
                                                    filler().low(), 
                                                   filler().high(),
                            tensor->mutable_data<T, CPUContext>());
    }
};

template <typename T, class Context>
class UniformFiller final : public Filler<T, Context> {
 public:
    UniformFiller(const TensorFiller& filler) : Filler<T, Context>(filler) {}
    void Fill(Tensor* tensor) override {
        math::RandomUniform<T, Context>(tensor->count(), 
                                         filler().low(), 
                                        filler().high(),
                    tensor->mutable_data<T, Context>());
    }
};

template <typename T, class Context>
class XavierFiller final : public Filler<T, Context> {
 public:
    XavierFiller(const TensorFiller& filler) : Filler<T, Context>(filler) {}
    void Fill(Tensor* tensor) override {
        int fan_in = tensor->count() / tensor->dim(0);
        int fan_out = tensor->count() / tensor->dim(1);
        float n = fan_in, scale = 3.0;
        if (filler().has_scale()) scale = filler().scale();
        if (filler().variance_norm() == TensorFiller_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / float(2);
        } else if (filler().variance_norm() == TensorFiller_VarianceNorm_FAN_OUT) {
            n = fan_out;
        }
        float limit = std::sqrt(scale / n);
        math::RandomUniform<T, Context>(tensor->count(), 
                                                 -limit, 
                                                  limit,
                    tensor->mutable_data<T, Context>());
    }
};

template <typename T, class Context>
class MSRAFiller final : public Filler <T, Context> {
 public:
    MSRAFiller(const TensorFiller& filler) : Filler<T, Context>(filler) {}
    void Fill(Tensor* tensor) override {
        int fan_in = tensor->count() / tensor->dim(0);
        int fan_out = tensor->count() / tensor->dim(1);
        float n = fan_in, scale = 2.0;
        if (filler().has_scale()) scale = filler().scale();
        if (filler().variance_norm() == TensorFiller_VarianceNorm_FAN_AVG) {
            n = (fan_in + fan_out) / float(2);
        } else if (filler().variance_norm() == TensorFiller_VarianceNorm_FAN_OUT) {
            n = fan_out;
        }
        float std = std::sqrt(scale / n);
        math::RandomNormal<T, Context>(tensor->count(),
                                              float(0), 
                                                   std, 
                   tensor->mutable_data<T, Context>());
    }
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

}    // namespace dragon

#endif    // DRAGON_UTILS_FILLER_H_