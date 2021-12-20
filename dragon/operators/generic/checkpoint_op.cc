#include "dragon/core/operator.h"

namespace dragon {

namespace {

class GradientMaker final : public GradientMakerBase {
 public:
  GRADIENT_MAKER_CTOR(GradientMaker);

  bool CopyArguments() const override {
    return false;
  }

  void Make() override {
    // Parse defs.
    string buffer_scope = "Buffer";
    vector<OperatorDef> defs;
    vector<OperatorDef*> tape_defs;
    for (const auto& arg : def().arg()) {
      if (arg.name() == "defs") {
        defs.resize(arg.strings_size());
        tape_defs.resize(arg.strings_size());
        for (int i = 0; i < arg.strings_size(); ++i) {
          defs[i].ParsePartialFromString(arg.strings(i));
          tape_defs[i] = &defs[i];
        }
      } else if (arg.name() == "buffer_scope") {
        buffer_scope = arg.s();
      }
    }

    // Restore intermediates.
    Map<string, string> outputs_to;
    Map<string, int> outputs_count;
    for (auto& op : defs) {
      vector<int> outputs_at(op.output_size(), -1);
      for (int i = 0; i < op.output_size(); ++i) {
        for (int j = 0; j < op.input_size(); ++j) {
          if (op.output(i) != op.input(j)) continue;
          outputs_at[i] = j;
          break;
        }
      }
      for (int i = 0; i < op.input_size(); ++i) {
        const auto& iter = outputs_to.find(op.input(i));
        if (iter != outputs_to.end()) op.set_input(i, iter->second);
      }
      for (int i = 0; i < op.output_size(); ++i) {
        if (outputs_at[i] >= 0) {
          op.set_output(i, op.input(outputs_at[i]));
        } else {
          const auto& output = op.output(i);
          outputs_count[output] += 1;
          *op.mutable_output(i) = outputs_to[output] =
              output + "_" + str::to(outputs_count[output]);
        }
      }
    }

    // Highlight outputs.
    for (const auto& output : def().output()) {
      outputs_to[outputs_to[output]] = output;
    }

    // Rewrite intermediates.
    int buffer_index = 0;
    const auto buffer_prefix = buffer_scope + "_";
    for (auto& op : defs) {
      for (int i = 0; i < op.input_size(); ++i) {
        const auto& iter = outputs_to.find(op.input(i));
        if (iter != outputs_to.end()) op.set_input(i, iter->second);
      }
      bool must_replay = false;
      for (int i = 0; i < op.output_size(); ++i) {
        const auto& iter = outputs_to.find(op.output(i));
        if (iter != outputs_to.end()) {
          op.set_output(i, iter->second);
          must_replay = str::startswith(op.output(i), buffer_prefix);
        } else {
          *op.mutable_output(i) = outputs_to[op.output(i)] =
              buffer_prefix + str::to(++buffer_index);
          must_replay = true;
        }
      }
      if (must_replay) {
        grad_defs_.push_back(op);
      }
    }

    // Create gradient defs.
    GradientTape tape;
    tape.CreateGradientDefs(
        tape_defs, {def().output().begin(), def().output().end()}, {});
    outputs_to.clear();
    const auto grad_prefix = def().name() + "/";
    for (const auto& grad_op : tape.def().op()) {
      if (grad_op.type() == "GradientFill") continue;
      grad_defs_.push_back(grad_op);
      // Make intermediate gradients unique to match the SSA.
      auto& op = grad_defs_.back();
      for (int i = 0; i < op.input_size(); ++i) {
        const auto& iter = outputs_to.find(op.input(i));
        if (iter != outputs_to.end()) op.set_input(i, iter->second);
      }
      for (int i = 0; i < op.output_size(); ++i) {
        const auto& output = op.output(i);
        if (str::startswith(output, buffer_prefix) &&
            str::find(output, "_grad")) {
          op.set_output(i, outputs_to[output] = grad_prefix + output);
        }
      }
    }

    // Create output gradients.
    for (int i = 0; i < def().input_size(); ++i) {
      grad_inputs_[i] = def().input(i) + "_grad";
    }
  }
};

} // namespace

REGISTER_GRADIENT(Checkpoint, GradientMaker);

} // namespace dragon
