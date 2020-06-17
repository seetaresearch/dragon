#include "dragon/core/graph_gradient.h"
#include "dragon/core/operator.h"

namespace dragon {

bool GraphGradientMaker::CheckGrad(
    const OperatorDef& forward_op,
    const Set<string>& targets,
    vector<pair<string, int>>& gen_grads) {
  if (NoGradientRegistry()->Has(forward_op.type())) {
    for (auto& input : forward_op.input())
      blacklist_set_.insert(input);
    return true;
  }
  for (int i = 0; i < forward_op.output_size(); ++i) {
    const auto& output = forward_op.output(i);
    if (!inputs_to_grads_.count(output)) {
      if (blacklist_set_.count(output)) return true;
      if (targets.count(output)) {
        // Consider to generate virtual gradient for targets
        gen_grads.push_back({output, i});
        inputs_to_grads_[output] = output + "_grad";
      } else if (forward_op.output_size() == 1) {
        return true; // We can skip this op, obviously
      }
    }
  }
  // Pass check, even if missing some grads
  return false;
}

void GraphGradientMaker::Make(
    const vector<OperatorDef*>& forward_ops,
    const vector<string>& targets,
    const vector<string>& input_grads,
    GraphDef& backward_def) {
  Map<string, int> inputs_count, grads_count;
  Set<string> all_split_grads, targets_set;
  Map<string, string> targets_to_grads;
  // PLAY for the forward
  for (auto* op : forward_ops) {
    if (NoGradientRegistry()->Has(op->type())) continue;
    for (const auto& input : op->input()) {
      bool input_in_outputs = false;
      for (auto& output : op->output())
        if (output == input) {
          input_in_outputs = true;
          break;
        }
      // Avoid to count the duplicate input(i.e. the in-place output)
      if (!input_in_outputs) inputs_count[input]++;
    }
  }
  // PLAY for the backward
  for (int i = 0; i < targets.size(); ++i) {
    // Set the gradient of targets
    if (i < input_grads.size()) {
      inputs_to_grads_[targets[i]] = input_grads[i];
    }
    targets_set.insert(targets[i]);
  }
  for (int i = (int)forward_ops.size() - 1; i >= 0; --i) {
    // Collect inputs and outputs, generate raw gradient ops
    const OperatorDef& op = *forward_ops[i];
    vector<pair<string, int>> gen_grads;
    bool is_skip = CheckGrad(op, targets_set, gen_grads);
    vector<string> g_outputs;
    for (auto& output : op.output()) {
      string g_output = "";
      if (inputs_to_grads_.count(output) > 0)
        g_output = inputs_to_grads_[output];
      g_outputs.emplace_back(g_output);
    }
    auto grad = MakeGradientForOp(op, g_outputs);

    // Process the raw gradient ops
    vector<OperatorDef> gather_ops;
    for (auto& grad_op : grad.ops) {
      // Set op name
      if (!grad_op.has_name()) grad_op.set_name(GetOperatorName());
      // Split and gather gradients for multi-used input
      for (int i = 0; i < grad_op.output_size(); ++i) {
        auto* output = grad_op.mutable_output(i);
        int original_idx = -1;
        for (int j = 0; j < grad.g_inputs.size(); ++j)
          if (grad_op.output(i) == grad.g_inputs[j]) original_idx = j;
        // Ignore unused && in-placee GI
        if (original_idx == -1) continue;
        bool output_in_inputs = false;
        for (const auto& input : grad_op.input())
          if (grad_op.output(i) == input) output_in_inputs = true;
        if (output_in_inputs) continue;
        // Found a split branch
        const auto& original_name = op.input(original_idx);
        if (inputs_count[original_name] > 1) {
          // Split
          auto split_name =
              *output + "_autosplit_" + str::to(grads_count[*output]++);
          if (!is_skip) all_split_grads.insert(split_name);
          // Gather
          if (grads_count[*output] == inputs_count[original_name]) {
            OperatorDef gather_op;
            gather_op.set_name(GetOperatorName());
            gather_op.set_type("GradientGather");
            gather_op.add_output(*output);
            if (grad_op.has_device_option()) {
              gather_op.mutable_device_option()->CopyFrom(
                  grad_op.device_option());
            }
            for (int j = 0; j < grads_count[*output]; j++) {
              auto key = *output + "_autosplit_" + str::to(j);
              if (all_split_grads.count(key)) gather_op.add_input(key);
            }
            gather_ops.push_back(gather_op);
          }
          *output = split_name;
        }
      }
    }

    // Now, append the required ops
    if (!is_skip) {
      // 1) GradientGenerateOp
      if (gen_grads.size() > 0) {
        vector<string> op_inputs, op_outputs;
        Argument arg_defaults;
        arg_defaults.set_name("defaults");
        for (auto& gen_grad : gen_grads) {
          op_inputs.push_back(gen_grad.first);
          op_outputs.emplace_back(gen_grad.first + "_grad");
          arg_defaults.add_floats(grad.defaults[gen_grad.second]);
        }
        auto generate_op = MakeOperatorDef(
            "GradientGenerate",
            GetOperatorName(),
            op_inputs,
            op_outputs,
            vector<Argument>({arg_defaults}));
        if (op.has_device_option()) {
          generate_op.mutable_device_option()->CopyFrom(op.device_option());
        }
        backward_def.add_op()->CopyFrom(generate_op);
      }
      // 2) GradientOp
      for (const auto& grad_op : grad.ops)
        backward_def.add_op()->CopyFrom(grad_op);
    }
    // 3) GradientGatherOp
    for (const auto& gather_op : gather_ops)
      backward_def.add_op()->CopyFrom(gather_op);

    // Done!
    if (!is_skip) {
      for (int i = 0; i < op.input_size(); ++i) {
        if (!grad.g_inputs[i].empty())
          inputs_to_grads_[op.input(i)] = grad.g_inputs[i];
      }
    }
  }
}

GraphDef GraphGradientMaker::Share(const GraphDef& input_def) {
  Set<int> invalid_ops;
  Map<string, int> ref_count;
  Map<string, pair<int, string>> ssa_map;
  // Count the refs for detecting leaf nodes
  for (int op_idx = 0; op_idx < input_def.op_size(); ++op_idx) {
    const auto& op = input_def.op(op_idx);
    if (!str::find(op.type(), "Gradient")) continue;
    if (op.type() == "GradientGather") {
      invalid_ops.insert(op_idx);
      if (ignored_grads_.count(op.output(0))) {
        for (const auto& input : op.input())
          ignored_grads_.insert(input);
        continue;
      } else {
        string head;
        for (const auto& input : op.input()) {
          if (!input.empty()) {
            if (head.empty()) head = input;
            ssa_map[input] = {op_idx, head};
          }
        }
      }
    }
    for (const auto& input : op.input())
      if (str::find(input, "grad")) ref_count[input] += 1;
  }

  // Decompose the <GradientGather> in SSA format
  GraphDef output_def(input_def);
  output_def.clear_op();
  for (int op_idx = 0; op_idx < input_def.op_size(); ++op_idx) {
    if (invalid_ops.count(op_idx)) continue;
    const auto& op = input_def.op(op_idx);
    output_def.add_op()->CopyFrom(op);
    if (!str::find(op.type(), "Gradient")) continue;
    for (const auto& output : op.output()) {
      const auto& find_iter = ssa_map.find(output);
      if (find_iter != ssa_map.end()) {
        const auto& gather_op = input_def.op(find_iter->second.first);
        auto acc_op(gather_op);
        acc_op.clear_input();
        if (output != find_iter->second.second) {
          acc_op.set_type("GradientAdd");
          // Make an in-place to avoid a new buffer
          acc_op.add_input(gather_op.output(0));
          const auto& ref_iter = ref_count.find(gather_op.output(0));
          if (ref_iter != ref_count.end()) ref_iter->second++;
        }
        acc_op.add_input(output);
        output_def.add_op()->CopyFrom(acc_op);
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
      const string& output = op->output(i);
      if (output.empty() || str::startswith(output, "/share/")) continue;
      if (ignored_grads_.count(output) > 0) {
        // Prune for non-trainable leafs
        *op->mutable_output(i) = "";
        continue;
      }
      if (hooked_grads_.empty()) {
        // Protection for leafs
        if (ref_count.count(output) == 0) continue;
      } else {
        // Protection for sources
        if (hooked_grads_.count(output) > 0) continue;
      }
      if (op->type() == "PythonPluginGradient") continue;
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
