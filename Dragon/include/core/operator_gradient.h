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

    virtual bool CopyDeviceOption() const { return true; }
    virtual bool CopyEngine() const { return true; }
    virtual bool CopyArguments() const { return true; }

    virtual Gradient Make() {
        auto new_defs = MakeDef();
        if (def.has_uid()) {
            // Attach the handle to name if having UID
            for (int i = 0; i < new_defs.size(); i++)
                new_defs[i].set_name(def.name());
        } else {
            // Otherwise, just put it into the arguments
            Argument arg;
            arg.set_name("handle");
            arg.set_s(def.name());
            for (int i = 0; i < new_defs.size(); i++)
                new_defs[i].add_arg()->CopyFrom(arg);
        }
        return Gradient(new_defs, g_inputs_, defaults());
    };

    virtual vector<OperatorDef> MakeDef() {
        return vector<OperatorDef>();
    }

    virtual vector<float> defaults() {
        return vector<float>(g_outputs_.size(), 1.f);
    }

    template <class... Args>
    static vector<OperatorDef> SingleDef(const Args& ... args) {
        return vector<OperatorDef> { MakeOperatorDef(args...) };
    }

    const string I(const int i) const {
        return i < def.input_size() ?
            def.input(i) : "NULL";
    }

    const string O(const int i) const {
        return i < def.output_size() ?
            def.output(i) : "NULL";
    }

    string GI(const int i) {
        if (i >= g_inputs_.size()) return "NULL";
        g_inputs_[i] = def.input(i) + "_grad";
        return g_inputs_[i];
    }

    const string GO(const int i) const {
        return i < g_outputs_.size() ?
            g_outputs_[i] : "NULL";
    }

 protected:
    const OperatorDef& def;
    vector<string> g_inputs_;
    const vector<string>& g_outputs_;
};

// Implemented in operator.cc
Gradient MakeGradientForOp(
    const OperatorDef&              op_def,
    const vector<string>&           g_outputs);

#define GRADIENT_MAKER_CTOR(name) \
    name(const OperatorDef& def, \
         const vector<string>& g_output) \
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

class SimpleGradientMaker final : public GradientMakerBase {
 public:
    /*!
     *              <SimpleMaker>
     *
     *    Inputs: X1, X2, ..., Xn, dY
     *    Outputs: dX1, dX2, ..., dXn
     *
     */
    GRADIENT_MAKER_CTOR(SimpleGradientMaker);
    vector<OperatorDef> MakeDef() override {
        vector<string> inputs, outputs;
        for (const auto& input : def.input())
            inputs.push_back(input);
        for (int i = 0; i < def.input_size(); ++i)
            outputs.push_back(GI(i));
        inputs.push_back(GO(0));
        return SingleDef(
            def.type() + "Gradient",
            "",
            inputs,
            outputs
        );
    }
};

class InplaceGradientMaker final : public GradientMakerBase {
 public:
    /*!
     *               <InplaceMaker>
     *
     *    Inputs:           Y, dY
     *    Outputs:           dX
     *
     */
    GRADIENT_MAKER_CTOR(InplaceGradientMaker);
    vector<OperatorDef> MakeDef() override {
        return SingleDef(
            def.type() + "Gradient",          /*!   OpType   */
            "",                               /*!   OpName   */
            vector<string>({ O(0), GO(0) }),  /*!   Inputs   */
            vector<string>({ GI(0) })         /*!   Outputs  */
        );
    }
};

}  // namespace

DECLARE_REGISTRY(
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&
);

DECLARE_REGISTRY(
    NoGradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&
);

// Defined in the operator.cc
#define REGISTER_GRADIENT(name, ...) \
    REGISTER_CLASS(GradientRegistry, name, __VA_ARGS__)

#define NO_GRADIENT(name) \
    REGISTER_GRADIENT(name, NoGradient); \
    REGISTER_CLASS(NoGradientRegistry, name, NoGradient)

}  // namespace dragon

#endif  // DRAGON_CORE_OPERATOR_GRADIENT_H_