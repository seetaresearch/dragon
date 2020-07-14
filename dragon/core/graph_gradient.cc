#include "dragon/core/graph_gradient.h"
#include "dragon/core/operator.h"

namespace dragon {

bool GraphGradientMaker::CheckGrad(
    const OperatorDef& op_def,
    const Set<string>& targets,
    vector<pair<string, int>>& gen_grads) {
  if (NoGradientRegistry()->Has(op_def.type())) {
    return true;
  }
  bool maybe_skip = false;
  for (int i = 0; i < op_def.output_size(); ++i) {
    const auto& output = op_def.output(i);
    if (!inputs_to_grads_.count(output)) {
      maybe_skip = true;
      if (targets.count(output)) {
        gen_grads.push_back({output, i});
        inputs_to_grads_[output] = output + "_grad";
      }
    }
  }
  return maybe_skip && gen_grads.empty();
}

void GraphGradientMaker::Make(
    const vector<OperatorDef*>& op_defs,
    const vector<string>& targets,
    const vector<string>& input_grads,
    GraphDef& graph_def) {
  Set<string> split_grads, targets_v2;
  Map<string, int> inputs_count, grads_count;

  // PLAY for the forward
  for (auto* op_def : op_defs) {
    if (NoGradientRegistry()->Has(op_def->type())) continue;
    for (const auto& input : op_def->input()) {
      bool input_in_outputs = false;
      for (auto& output : op_def->output())
        if (output == input) {
          input_in_outputs = true;
          break;
        }
      // Avoid to count the duplicate input (i.e. the in-place output)
      if (!input_in_outputs) inputs_count[input]++;
    }
  }

  // Set the gradient of targets
  for (int i = 0; i < targets.size(); ++i) {
    if (i < input_grads.size()) {
      inputs_to_grads_[targets[i]] = input_grads[i];
    }
    targets_v2.insert(targets[i]);
  }

  // PLAY for the backward
  for (int op_idx = (int)op_defs.size() - 1; op_idx >= 0; --op_idx) {
    const OperatorDef& op_def = *op_defs[op_idx];
    // Generate def by registered gradient maker
    vector<pair<string, int>> gen_grads;
    vector<string> grad_outputs;
    bool is_skip = CheckGrad(op_def, targets_v2, gen_grads);
    for (const auto& output : op_def.output()) {
      string grad_output = "";
      const auto& it = inputs_to_grads_.find(output);
      if (it != inputs_to_grads_.end()) grad_output = it->second;
      grad_outputs.push_back(grad_output);
    }
    auto pack = MakeGradientForOp(op_def, grad_outputs);
    // Split and gather gradient for multi-used inputs
    vector<OperatorDef> gather_defs;
    for (auto& grad_def : pack.grad_defs) {
      if (!grad_def.has_name()) {
        grad_def.set_name(GetOperatorName());
      }
      for (int i = 0; i < grad_def.output_size(); ++i) {
        const auto& grad_name = grad_def.output(i);
        int original_index = -1;
        for (int j = 0; j < pack.grad_inputs.size(); ++j) {
          if (grad_name == pack.grad_inputs[j]) {
            original_index = j;
          }
        }
        if (original_index == -1) continue;
        bool output_in_inputs = false;
        for (const auto& name : grad_def.input()) {
          if (grad_name == name) {
            output_in_inputs = true;
            break;
          }
        }
        if (output_in_inputs) continue;
        // Detect a split branch
        const auto& original_name = op_def.input(original_index);
        if (inputs_count[original_name] > 1) {
          auto grad_name_v2 =
              grad_name + "_autosplit_" + str::to(grads_count[grad_name]++);
          if (!is_skip) split_grads.insert(grad_name_v2);
          if (grads_count[grad_name] == inputs_count[original_name]) {
            auto gather_def = MakeOperatorDef(
                "GradientGather",
                GetOperatorName(),
                vector<string>({}),
                vector<string>({grad_name}));
            if (grad_def.has_device_option()) {
              gather_def.mutable_device_option()->CopyFrom(
                  grad_def.device_option());
            }
            for (int j = 0; j < grads_count[grad_name]; j++) {
              auto name = grad_name + "_autosplit_" + str::to(j);
              if (split_grads.count(name)) gather_def.add_input(name);
            }
            gather_defs.push_back(gather_def);
          }
          *grad_def.mutable_output(i) = grad_name_v2;
        }
      }
    }

    // Add defs
    if (!is_skip) {
      for (int i = 0; i < op_def.input_size(); ++i) {
        inputs_to_grads_[op_def.input(i)] = pack.grad_inputs[i];
      }
      // Add def for ``GradientGenerateOp``
      if (gen_grads.size() > 0) {
        vector<string> inputs, outputs;
        Argument arg_defaults;
        arg_defaults.set_name("defaults");
        for (auto& gen_grad : gen_grads) {
          inputs.push_back(gen_grad.first);
          outputs.emplace_back(gen_grad.first + "_grad");
          arg_defaults.add_floats(pack.defaults[gen_grad.second]);
        }
        auto generate_def = MakeOperatorDef(
            "GradientGenerate",
            GetOperatorName(),
            inputs,
            outputs,
            vector<Argument>({arg_defaults}));
        if (op_def.has_device_option()) {
          generate_def.mutable_device_option()->CopyFrom(
              op_def.device_option());
        }
        graph_def.add_op()->CopyFrom(generate_def);
      }
      // Add def for ``GenerateOp``
      for (const auto& grad_def : pack.grad_defs) {
        graph_def.add_op()->CopyFrom(grad_def);
      }
    }
    // Add def for ``GradientGatherOp``
    for (const auto& gather_def : gather_defs) {
      graph_def.add_op()->CopyFrom(gather_def);
    }
  }
}

GraphDef GraphGradientMaker::Share(const GraphDef& input_def) {
  Set<int> invalid_ops;
  Map<string, int> ref_count;
  Map<string, pair<int, string>> gather_map;

  for (int op_idx = 0; op_idx < input_def.op_size(); ++op_idx) {
    const auto& op = input_def.op(op_idx);
    if (!str::find(op.type(), "Gradient")) continue;
    // Flag the gathering gradients
    if (op.type() == "GradientGather") {
      invalid_ops.insert(op_idx);
      if (empty_grads_.count(op.output(0))) {
        for (const auto& input : op.input()) {
          empty_grads_.insert(input);
        }
        continue;
      } else {
        string first_input;
        for (const auto& input : op.input()) {
          if (!input.empty()) {
            if (first_input.empty()) first_input = input;
            gather_map[input] = {op_idx, first_input};
          }
        }
      }
    }
    // Count the references to detect leafs
    for (const auto& input : op.input()) {
      if (str::endswith(input, "_grad")) {
        ref_count[input] += 1;
      }
    }
  }

  // Decompose the <GradientGather> into <GradientAdd>
  // This trick accumulates the split to target right after computing,
  // which helps to reduce the total number of buffers.
  GraphDef output_def(input_def);
  output_def.clear_op();
  for (int op_idx = 0; op_idx < input_def.op_size(); ++op_idx) {
    if (invalid_ops.count(op_idx)) continue;
    const auto& op = input_def.op(op_idx);
    output_def.add_op()->CopyFrom(op);
    if (!str::find(op.type(), "Gradient")) continue;
    for (const auto& output : op.output()) {
      const auto& find_iter = gather_map.find(output);
      if (find_iter != gather_map.end()) {
        const auto& gather_op = input_def.op(find_iter->second.first);
        auto add_op(gather_op);
        add_op.clear_input();
        if (output != find_iter->second.second) {
          add_op.set_type("GradientAdd");
          // Make an in-place to avoid a new buffer
          add_op.add_input(gather_op.output(0));
          const auto& ref_iter = ref_count.find(gather_op.output(0));
          if (ref_iter != ref_count.end()) ref_iter->second++;
        }
        add_op.add_input(output);
        output_def.add_op()->CopyFrom(add_op);
      }
    }
  }

  // Prepare the pool
  int buffer_idx = 0;
  std::deque<string> pool;
  Map<string, string> grad_to_buffer;
  auto get_buffer = [&]() mutable {
    if (pool.empty()) {
      return "/share/buffer/grad:" + str::to(buffer_idx++);
    } else {
      /*!
       * LIFO is more memory efficient than FIFO usually,
       * Because the larger gradients will bring out later.
       *
       * Memory distribution turns out to be uniform,
       * if the early temporary tensors are selected prior.
       */
      auto buffer = pool.back();
      pool.pop_back();
      return buffer;
    }
  };

  for (int op_idx = 0; op_idx < output_def.op_size(); ++op_idx) {
    auto* op = output_def.mutable_op(op_idx);
    // Ignore the non-gradient ops
    if (!str::find(op->type(), "Gradient")) continue;
    // Check if output is an alias of input
    vec32_t inplace_flags;
    for (int i = 0; i < op->output_size(); ++i) {
      int flag = -1;
      for (int j = 0; j < op->input_size(); ++j)
        if (op->output(i) == op->input(j)) {
          flag = j;
          break;
        }
      inplace_flags.emplace_back(flag);
    }
    // Besides, we need to collect the dead buffers
    // Reuse them when current operator is done
    vector<string> dead_buffers;
    // Rewrite input gradients
    for (int i = 0; i < op->input_size(); ++i) {
      const string& input = op->input(i);
      if (ref_count.count(input) > 0) {
        ref_count[input] -= 1; // Decref
        if (grad_to_buffer.count(input) == 0) continue;
        string new_input = grad_to_buffer[input];
        if (ref_count[input] == 0) {
          dead_buffers.emplace_back(new_input);
        }
        *op->mutable_input(i) = new_input;
      }
    }
    // Rewrite output gradients
    for (int i = 0; i < op->output_size(); ++i) {
      if (str::startswith(op->type(), "Python")) continue;
      const string& output = op->output(i);
      if (output.empty() || str::startswith(output, "/share/buffer")) continue;
      if (empty_grads_.count(output) > 0) {
        *op->mutable_output(i) = "";
        continue;
      }
      // Protection for leafs
      if (ref_count.count(output) == 0) continue;
      // Protection for sources and leafs
      if (retained_grads_.count(output) > 0) continue;
      string new_output = output;
      if (inplace_flags[i] >= 0) {
        new_output = op->input(inplace_flags[i]);
      } else {
        grad_to_buffer[output] = new_output = get_buffer();
      }
      *op->mutable_output(i) = new_output;
    }
    // Update the pool
    for (auto& buffer : dead_buffers) {
      pool.emplace_back(buffer);
    }
  }

  return output_def;
}

} // namespace dragon
