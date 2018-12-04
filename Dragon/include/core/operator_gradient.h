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

#ifndef DRAGON_CORE_OPERATOR_GRADIENT_H_
#define DRAGON_CORE_OPERATOR_GRADIENT_H_

#include "core/common.h"
#include "core/registry.h"
#include "core/operator.h"
#include "utils/proto_utils.h"

namespace dragon {

struct Gradient {
    vector<OperatorDef> ops;
    vector<string> g_inputs;
    vector<float> defaults;
    Gradient(
        const vector<OperatorDef>&   ops,
        const vector<string>&        g_inputs,
        const vector<float>&         defaults)
        : ops(ops), g_inputs(g_inputs), defaults(defaults) {}
};

class GradientMakerBase {
 public:
    GradientMakerBase(
        const OperatorDef&          def,
        const vector<string>&       g_outputs)
        : def(def), g_outputs_(g_outputs),
          g_inputs_(def.input_size()) {}
    virtual ~GradientMakerBase() {}

    inline virtual bool CopyDeviceOption() const { return true; }
    inline virtual bool CopyEngine() const { return true; }
    inline virtual bool CopyArguments() const { return true; }

    inline virtual Gradient Make() {
        vector<OperatorDef> new_defs = MakeDefs();
        Argument anchor;
        anchor.set_name("anchor"); anchor.set_s(def.name());
        for (int i = 0; i < new_defs.size(); i++)
            new_defs[i].add_arg()->CopyFrom(anchor);
        return Gradient(new_defs, g_inputs_, DefaultValues());
    };

    virtual inline vector<OperatorDef> MakeDefs() {
        NOT_IMPLEMENTED;
        return vector<OperatorDef>();
    }

    virtual inline vector<float> DefaultValues() {
        return vector<float>(g_outputs_.size(), 1.f);
    }

    template <class... Args>
    inline static vector<OperatorDef> SingleDef(const Args& ... args) {
        return vector<OperatorDef> { MakeOperatorDef(args...) };
    }

    inline string I(const int i) { return def.input(i); }
    inline string O(const int i) { return def.output(i); }

    inline string GI(const int i) {
        if (i >= g_inputs_.size()) return "ignore";
        g_inputs_[i] = def.input(i) + "_grad";
        return g_inputs_[i];
    }
    inline string GO(const int i) { return g_outputs_[i]; }

 protected:
    const OperatorDef& def;
    vector<string> g_inputs_;
    const vector<string>& g_outputs_;
};

// Implemented in operator.cc
Gradient MakeGradientForOp(
    const OperatorDef&              op_def,
    const vector<string>&           g_outputs);

# define GRADIENT_MAKER_CTOR(name) \
    name(const OperatorDef& def, const vector<string>& g_output) \
        : GradientMakerBase(def, g_output) {}

class NoGradient : public GradientMakerBase {
 public:
    GRADIENT_MAKER_CTOR(NoGradient);
    vector<OperatorDef> MakeDefs() override {
        return vector<OperatorDef>();
    }
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

// Defined in the operator.cc
#define REGISTER_GRADIENT(name, ...) \
    REGISTER_CLASS(GradientRegistry, name, __VA_ARGS__)

#define NO_GRADIENT(name) \
    REGISTER_GRADIENT(name, NoGradient); \
    REGISTER_CLASS(NoGradientRegistry, name, NoGradient)

}  // namespace dragon

#endif  // DRAGON_CORE_OPERATOR_GRADIENT_H_