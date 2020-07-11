/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *    <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_UTILS_FILLER_H_
#define DRAGON_UTILS_FILLER_H_

#include "dragon/core/registry.h"
#include "dragon/utils/math_functions.h"

namespace dragon {

template <typename T, class Context>
class Filler {
 public:
  explicit Filler(const FillerInfo& info) : info_(info) {}
  virtual ~Filler() {}

  virtual void Fill(Tensor* X, Context* ctx) = 0;

  const FillerInfo& info() {
    return info_;
  }

 protected:
  FillerInfo info_;
};

template <typename T, class Context>
class ConstantFiller final : public Filler<T, Context> {
 public:
  explicit ConstantFiller(const FillerInfo& info) : Filler<T, Context>(info) {}

  void Fill(Tensor* X, Context* ctx) override {
    math::Set(
        X->count(),
        cast::to<T>(info().value()),
        X->mutable_data<T, Context>(),
        ctx);
  }

 protected:
  using Filler<T, Context>::info;
};

template <typename T, class Context>
class NormalFiller final : public Filler<T, Context> {
 public:
  explicit NormalFiller(const FillerInfo& info) : Filler<T, Context>(info) {}

  void Fill(Tensor* X, Context* ctx) override {
    math::RandomNormal(
        X->count(),
        info().mean(),
        info().std(),
        X->mutable_data<T, Context>(),
        ctx);
  }

 protected:
  using Filler<T, Context>::info;
};

template <typename T, class Context>
class TruncatedNormalFiller final : public Filler<T, Context> {
 public:
  explicit TruncatedNormalFiller(const FillerInfo& info)
      : Filler<T, Context>(info) {}

  void Fill(Tensor* X, Context* /* ctx */) override {
    CPUContext ctx; // Enforce the cpu implementation
    math::TruncatedNormal(
        X->count(),
        info().mean(),
        info().std(),
        info().low(),
        info().high(),
        X->mutable_data<T, CPUContext>(),
        &ctx);
  }

 protected:
  using Filler<T, Context>::info;
};

template <typename T, class Context>
class UniformFiller final : public Filler<T, Context> {
 public:
  explicit UniformFiller(const FillerInfo& info) : Filler<T, Context>(info) {}

  void Fill(Tensor* X, Context* ctx) override {
    math::RandomUniform(
        X->count(),
        info().low(),
        info().high(),
        X->mutable_data<T, Context>(),
        ctx);
  }

 protected:
  using Filler<T, Context>::info;
};

template <typename T, class Context>
class GlorotUniformFiller final : public Filler<T, Context> {
 public:
  explicit GlorotUniformFiller(const FillerInfo& info)
      : Filler<T, Context>(info) {}

  void Fill(Tensor* X, Context* ctx) override {
    auto fan_in = X->count() / X->dim(0);
    auto fan_out = X->count() / X->dim(1);
    float n = (float)fan_in, scale = 3.f;
    if (info().has_scale()) scale = info().scale();
    if (info().variance_norm() == FillerInfo_VarianceNorm_FAN_AVG) {
      n = (fan_in + fan_out) / 2.f;
    } else if (info().variance_norm() == FillerInfo_VarianceNorm_FAN_OUT) {
      n = (float)fan_out;
    }
    float limit = std::sqrt(scale / n);
    math::RandomUniform(
        X->count(), -limit, limit, X->mutable_data<T, Context>(), ctx);
  }

 protected:
  using Filler<T, Context>::info;
};

template <typename T, class Context>
class GlorotNormalFiller final : public Filler<T, Context> {
 public:
  explicit GlorotNormalFiller(const FillerInfo& info)
      : Filler<T, Context>(info) {}

  void Fill(Tensor* X, Context* ctx) override {
    auto fan_in = X->count() / X->dim(0);
    auto fan_out = X->count() / X->dim(1);
    float n = (float)fan_in, scale = 2.f;
    if (info().has_scale()) scale = info().scale();
    if (info().variance_norm() == FillerInfo_VarianceNorm_FAN_AVG) {
      n = (fan_in + fan_out) / 2.f;
    } else if (info().variance_norm() == FillerInfo_VarianceNorm_FAN_OUT) {
      n = (float)fan_out;
    }
    float std = std::sqrt(scale / n);
    math::RandomNormal(
        X->count(), 0.f, std, X->mutable_data<T, Context>(), ctx);
  }

 protected:
  using Filler<T, Context>::info;
};

template <typename T, class Context>
Filler<T, Context>* CreateFiller(const FillerInfo& info) {
  const string& type = info.type();
  if (type == "constant") {
    return new ConstantFiller<T, Context>(info);
  } else if (type == "uniform") {
    return new UniformFiller<T, Context>(info);
  } else if (type == "normal") {
    return new NormalFiller<T, Context>(info);
  } else if (type == "truncated_normal") {
    return new TruncatedNormalFiller<T, Context>(info);
  } else if (type == "glorot_uniform" || type == "xavier") {
    return new GlorotUniformFiller<T, Context>(info);
  } else if (type == "glorot_normal" || type == "msra") {
    return new GlorotNormalFiller<T, Context>(info);
  }
  return new ConstantFiller<T, Context>(info);
}

} // namespace dragon

#endif // DRAGON_UTILS_FILLER_H_
