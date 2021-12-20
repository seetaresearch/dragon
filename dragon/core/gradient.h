/*!
 * Copyright (c) 2017-present, SeetaTech, Co.,Ltd.
 *
 * Licensed under the BSD 2-Clause License.
 * You should have received a copy of the BSD 2-Clause License
 * along with the software. If not, See,
 *
 *     <https://opensource.org/licenses/BSD-2-Clause>
 *
 * ------------------------------------------------------------
 */

#ifndef DRAGON_CORE_GRADIENT_H_
#define DRAGON_CORE_GRADIENT_H_

#include "dragon/core/common.h"
#include "dragon/core/registry.h"
#include "dragon/utils/proto_utils.h"

namespace dragon {

class GradientMakerBase {
 public:
  GradientMakerBase(const OperatorDef& def, const vector<string>& grad_outputs)
      : def_(def),
        grad_outputs_(grad_outputs),
        grad_inputs_(def.input_size()) {}

  virtual ~GradientMakerBase() {}

  virtual bool CopyArguments() const {
    return true;
  }

  virtual bool CopyDeviceOption() const {
    return true;
  }

  virtual bool CopyEngine() const {
    return true;
  }

  virtual void Make() {
    CreateGradientDefs();
    string cache_key;
    if (!def_.arg().empty()) {
      const auto& arg = *(def_.arg().end() - 1);
      if (arg.name() == "cache_key") cache_key = arg.s();
    }
    Argument new_arg;
    new_arg.set_name("name");
    new_arg.set_s(def_.name());
    for (auto& grad_def : grad_defs_) {
      if (CopyDeviceOption() && def_.has_device_option()) {
        grad_def.mutable_device_option()->CopyFrom(def_.device_option());
      }
      if (CopyArguments() && !def_.arg().empty()) {
        grad_def.mutable_arg()->MergeFrom(def_.arg());
        if (!cache_key.empty()) grad_def.mutable_arg()->RemoveLast();
      }
      grad_def.add_arg()->CopyFrom(new_arg);
    }
    if (!cache_key.empty()) {
      cache_key += "/grad";
      new_arg.set_name("cache_key");
      for (int i = 0; i < grad_defs_.size(); ++i) {
        new_arg.set_s(cache_key + (i > 0 ? ("/" + str::to(i)) : ""));
        grad_defs_[i].add_arg()->CopyFrom(new_arg);
      }
    }
  };

  virtual void CreateGradientDefs() {}

  template <class... Args>
  void AddGradientDef(const Args&... args) {
    grad_defs_.emplace_back(CreateOperatorDef(args...));
  }

  const string I(const int i) const {
    return i < int(def_.input_size()) ? def_.input(i) : "";
  }

  const string O(const int i) const {
    return i < int(def_.output_size()) ? def_.output(i) : "";
  }

  string GI(const int i) {
    if (i >= int(grad_inputs_.size())) return "";
    grad_inputs_[i] = def_.input(i) + "_grad";
    return grad_inputs_[i];
  }

  const string GO(const int i) const {
    return i < int(grad_outputs_.size()) ? grad_outputs_[i] : "";
  }

  const OperatorDef& def() const {
    return def_;
  }

  vector<OperatorDef>& grad_defs() {
    return grad_defs_;
  }

  vector<string>& grad_inputs() {
    return grad_inputs_;
  }

  virtual vector<float> grad_defaults() {
    return vector<float>(grad_outputs_.size(), 1.f);
  }

 protected:
  const OperatorDef& def_;
  vector<OperatorDef> grad_defs_;
  const vector<string>& grad_outputs_;
  vector<string> grad_inputs_;
};

#define GRADIENT_MAKER_CTOR(name)                                  \
  name(const OperatorDef& def, const vector<string>& grad_outputs) \
      : GradientMakerBase(def, grad_outputs) {}

class NoGradient : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(NoGradient);
};

namespace {

class GenericGradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GenericGradientMaker);
  void CreateGradientDefs() override {
    /*!
     * X1, X2, ..., Xn, dY1, dY2, ..., dYm
     * dX1, dX2, ..., dXn
     */
    vector<string> inputs({def().input().begin(), def().input().end()});
    vector<string> outputs;
    for (int i = 0; i < def().output_size(); ++i) {
      inputs.emplace_back(GO(i));
    }
    for (int i = 0; i < def().input_size(); ++i) {
      outputs.emplace_back(GI(i));
    }
    AddGradientDef(def().type() + "Gradient", "", inputs, outputs);
  }
};

class SimpleGradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(SimpleGradientMaker);
  void CreateGradientDefs() override {
    /*!
     * dY1, dY2, ..., dYm
     * dX1, dX2, ..., dXn
     */
    vector<string> inputs, outputs;
    for (int i = 0; i < def().output_size(); ++i) {
      inputs.emplace_back(GO(i));
    }
    for (int i = 0; i < def().input_size(); ++i) {
      outputs.emplace_back(GI(i));
    }
    AddGradientDef(def().type() + "Gradient", "", inputs, outputs);
  }
};

class InplaceGradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(InplaceGradientMaker);
  void CreateGradientDefs() override {
    /*!
     * Y, dY
     * dX
     */
    AddGradientDef(
        def().type() + "Gradient",
        "",
        vector<string>({O(0), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

class DRAGON_API GradientTape {
 public:
  GradientTape() {}
  GradientTape(const GraphDef& def) : def_(def) {}
  GradientTape(
      const vector<OperatorDef*>& op_defs,
      const vector<string>& targets,
      const vector<string>& grad_targets) {
    CreateGradientDefs(op_defs, targets, grad_targets);
  }

  /*! \brief Create gradient defs */
  void CreateGradientDefs(
      const vector<OperatorDef*>& op_defs,
      const vector<string>& targets,
      const vector<string>& grad_targets);

  /*! \brief Optimize gradient computations */
  void Optimize(const vector<string>& sources = vector<string>());

  /*! \brief Return gradient defs */
  const GraphDef& def() {
    if (optimized_def_.op_size() > 0) {
      return optimized_def_;
    }
    return def_;
  }

 private:
  GraphDef def_;
  GraphDef optimized_def_;
};

DECLARE_REGISTRY(
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&);

DECLARE_REGISTRY(
    NoGradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&);

#define REGISTER_GRADIENT(name, ...) \
  REGISTER_CLASS(GradientRegistry, name, __VA_ARGS__)

#define NO_GRADIENT(name)              \
  REGISTER_GRADIENT(name, NoGradient); \
  REGISTER_CLASS(NoGradientRegistry, name, NoGradient)

} // namespace dragon

#endif // DRAGON_CORE_GRADIENT_H_
