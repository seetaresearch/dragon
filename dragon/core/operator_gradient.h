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

#ifndef DRAGON_CORE_OPERATOR_GRADIENT_H_
#define DRAGON_CORE_OPERATOR_GRADIENT_H_

#include "dragon/core/common.h"
#include "dragon/core/operator.h"
#include "dragon/core/registry.h"
#include "dragon/utils/proto_utils.h"

namespace dragon {

struct GradientPack {
  GradientPack(
      const vector<OperatorDef>& grad_defs,
      const vector<string>& grad_inputs,
      const vector<float>& defaults)
      : grad_defs(grad_defs), grad_inputs(grad_inputs), defaults(defaults) {}

  vector<OperatorDef> grad_defs;
  vector<string> grad_inputs;
  vector<float> defaults;
};

class GradientMakerBase {
 public:
  GradientMakerBase(const OperatorDef& def, const vector<string>& grad_outputs)
      : def(def), grad_inputs_(def.input_size()), grad_outputs_(grad_outputs) {}

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

  virtual GradientPack Make() {
    auto new_defs = MakeDef();
    if (def.has_cache_key()) {
      // Attach the handle to name if having cache key
      for (size_t i = 0; i < new_defs.size(); i++) {
        new_defs[i].set_name(def.name());
      }
    } else {
      // Otherwise, just put it into the arguments
      Argument arg;
      arg.set_name("handle");
      arg.set_s(def.name());
      for (size_t i = 0; i < new_defs.size(); i++) {
        new_defs[i].add_arg()->CopyFrom(arg);
      }
    }
    return GradientPack(new_defs, grad_inputs_, defaults());
  };

  virtual vector<OperatorDef> MakeDef() {
    return vector<OperatorDef>();
  }

  template <class... Args>
  static vector<OperatorDef> SingleDef(const Args&... args) {
    return vector<OperatorDef>{MakeOperatorDef(args...)};
  }

  const string I(const int i) const {
    return i < int(def.input_size()) ? def.input(i) : "";
  }

  const string O(const int i) const {
    return i < int(def.output_size()) ? def.output(i) : "";
  }

  string GI(const int i) {
    if (i >= int(grad_inputs_.size())) return "";
    grad_inputs_[i] = def.input(i) + "_grad";
    return grad_inputs_[i];
  }

  const string GO(const int i) const {
    return i < int(grad_outputs_.size()) ? grad_outputs_[i] : "";
  }

  virtual vector<float> defaults() {
    return vector<float>(grad_outputs_.size(), 1.f);
  }

 protected:
  const OperatorDef& def;
  vector<string> grad_inputs_;
  const vector<string>& grad_outputs_;
};

DRAGON_API GradientPack
MakeGradientForOp(const OperatorDef& op_def, const vector<string>& g_outputs);

#define GRADIENT_MAKER_CTOR(name)                              \
  name(const OperatorDef& def, const vector<string>& g_output) \
      : GradientMakerBase(def, g_output) {}

class NoGradient : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(NoGradient);
  vector<OperatorDef> MakeDef() override {
    return vector<OperatorDef>();
  }
};

namespace {

// Here we define some common gradient makers
// Reuse them to make the codes cleaner

class GenericGradientMaker final : public GradientMakerBase {
 public:
  /*!
   * Inputs: X1, X2, ..., Xn, dY1, dY2, ..., dYm
   * Outputs: dX1, dX2, ..., dXn
   */
  GRADIENT_MAKER_CTOR(GenericGradientMaker);
  vector<OperatorDef> MakeDef() override {
    vector<string> inputs, outputs;
    for (const auto& input : def.input())
      inputs.push_back(input);
    for (int i = 0; i < def.output_size(); ++i)
      inputs.push_back(GO(i));
    for (int i = 0; i < def.input_size(); ++i)
      outputs.push_back(GI(i));
    return SingleDef(def.type() + "Gradient", "", inputs, outputs);
  }
};

class SimpleGradientMaker final : public GradientMakerBase {
 public:
  /*!
   * Inputs: dY1, dY2, ..., dYm
   * Outputs: dX1, dX2, ..., dXn
   */
  GRADIENT_MAKER_CTOR(SimpleGradientMaker);
  vector<OperatorDef> MakeDef() override {
    vector<string> inputs, outputs;
    for (int i = 0; i < def.output_size(); ++i)
      inputs.push_back(GO(i));
    for (int i = 0; i < def.input_size(); ++i)
      outputs.push_back(GI(i));
    return SingleDef(def.type() + "Gradient", "", inputs, outputs);
  }
};

class InplaceGradientMaker final : public GradientMakerBase {
 public:
  /*!
   * Inputs: Y, dY
   * Outputs: dX
   */
  GRADIENT_MAKER_CTOR(InplaceGradientMaker);
  vector<OperatorDef> MakeDef() override {
    return SingleDef(
        def.type() + "Gradient",
        "",
        vector<string>({O(0), GO(0)}),
        vector<string>({GI(0)}));
  }
};

} // namespace

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

// Defined in the operator.cc
#define REGISTER_GRADIENT(name, ...) \
  REGISTER_CLASS(GradientRegistry, name, __VA_ARGS__)

#define NO_GRADIENT(name)              \
  REGISTER_GRADIENT(name, NoGradient); \
  REGISTER_CLASS(NoGradientRegistry, name, NoGradient)

} // namespace dragon

#endif // DRAGON_CORE_OPERATOR_GRADIENT_H_
