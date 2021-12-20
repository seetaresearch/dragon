#include "dragon/core/gradient.h"

namespace dragon {

void GradientTape::CreateGradientDefs(
    const vector<OperatorDef*>& op_defs,
    const vector<string>& targets,
    const vector<string>& grad_targets) {
  Map<string, string> sources_to_grads;
  Map<string, string> targets_to_grads;
  Map<string, int> inputs_count;
  Map<string, int> splits_count;

  def_.clear_op();

  // Function to check if grad op can be removed.
  auto IsNoGradient = [&](const OperatorDef& op,
                          vector<pair<string, int>>& init_grads) mutable {
    if (NoGradientRegistry()->Has(op.type())) {
      return true;
    }
    bool maybe_skip = false;
    for (int i = 0; i < op.output_size(); ++i) {
      const auto& y = op.output(i);
      if (!sources_to_grads.count(y)) {
        maybe_skip = true;
        if (targets_to_grads.count(y)) {
          init_grads.push_back({y, i});
          sources_to_grads[y] = y + "_grad";
        }
      }
    }
    return maybe_skip && init_grads.empty() && op.output_size() == 1;
  };

  // Set the gradient of targets.
  for (int i = 0; i < targets.size(); ++i) {
    targets_to_grads[targets[i]] =
        i < grad_targets.size() ? grad_targets[i] : "";
  }

  // PLAY for the forward.
  for (auto* op : op_defs) {
    if (NoGradientRegistry()->Has(op->type())) continue;
    for (const auto& x : op->input()) {
      if (std::find(op->output().begin(), op->output().end(), x) ==
          op->output().end()) {
        inputs_count[x]++;
      }
    }
  }

  // PLAY for the backward.
  for (auto op_iter = op_defs.rbegin(); op_iter != op_defs.rend(); op_iter++) {
    const auto& op = *op_iter;
    vector<pair<string, int>> init_grads;
    if (IsNoGradient(*op, init_grads)) continue;
    CHECK(GradientRegistry()->Has(op->type()))
        << "\nNo gradient maker registered for " << op->type() << ".";

    // Create gradient defs.
    vector<string> grad_ys;
    for (const auto& y : op->output()) {
      const auto& iter = sources_to_grads.find(y);
      grad_ys.emplace_back(iter != sources_to_grads.end() ? iter->second : "");
    }
    unique_ptr<GradientMakerBase> maker(
        GradientRegistry()->Create(op->type(), *op, grad_ys));
    maker->Make();

    // Update input gradients.
    for (int i = 0; i < op->input_size(); ++i) {
      sources_to_grads[op->input(i)] = maker->grad_inputs()[i];
    }

    // Rewrite output gradients.
    vector<OperatorDef> grad_defs;
    for (auto& grad_op : maker->grad_defs()) {
      for (int i = 0; i < grad_op.output_size(); ++i) {
        if (grad_op.output(i).empty()) continue;
        const auto grad_x = grad_op.output(i);
        const auto& grad_ys = grad_op.input();
        if (std::find(grad_ys.begin(), grad_ys.end(), grad_x) !=
            grad_ys.end()) {
          continue;
        }
        int x_index = -1;
        for (int j = 0; j < maker->grad_inputs().size(); ++j) {
          if (grad_x == maker->grad_inputs()[j]) x_index = j;
        }
        if (x_index == -1) continue;
        const auto& count_iter = inputs_count.find(op->input(x_index));
        if (count_iter->second <= 1) continue;
        auto split_prefix = grad_x + "_split_";
        auto split_index = splits_count[grad_x]++;
        grad_op.set_output(i, split_prefix + str::to(split_index));
        if (split_index + 1 != count_iter->second) continue;
        grad_defs.emplace_back(CreateOperatorDef(
            "GradientGather",
            "",
            vector<string>({}),
            vector<string>({grad_x}),
            vector<Argument>(),
            grad_op.device_option()));
        for (int j = 0; j < count_iter->second; ++j) {
          grad_defs.back().add_input(split_prefix + str::to(j));
        }
      }
    }

    if (init_grads.size() > 0) {
      Argument values;
      values.set_name("values");
      vector<string> inputs, outputs;
      auto fills = maker->grad_defaults();
      for (auto& iter : init_grads) {
        const auto& grad = targets_to_grads[iter.first];
        inputs.emplace_back(grad.empty() ? iter.first : grad);
        outputs.emplace_back(iter.first + "_grad");
        values.add_floats(grad.empty() ? fills[iter.second] : -100.f);
      }
      def_.add_op()->CopyFrom(CreateOperatorDef(
          "GradientFill",
          "",
          inputs,
          outputs,
          vector<Argument>({values}),
          op->device_option()));
    }
    for (const auto& grad_op : maker->grad_defs()) {
      def_.add_op()->CopyFrom(grad_op);
    }
    for (const auto& grad_op : grad_defs) {
      def_.add_op()->CopyFrom(grad_op);
    }
  }
}

void GradientTape::Optimize(const vector<string>& sources) {
  Set<size_t> noop_indices;
  Set<string> retained_grads;
  Map<string, int> inputs_count;
  Map<string, int> outputs_count;
  Map<string, string> outputs_to;
  Map<string, pair<int, string>> splits;

  optimized_def_ = def_;
  optimized_def_.clear_op();

  // Initialize retained grads before optimization.
  for (const auto& input : sources) {
    retained_grads.insert(input + "_grad");
  }

  for (size_t op_index = 0; op_index < def_.op_size(); ++op_index) {
    const auto& op = def_.op(op_index);
    if (!str::find(op.type(), "Gradient")) continue;
    if (op.type() == "GradientGather") {
      noop_indices.insert(op_index); // Count noops.
    }
    for (const auto& input : op.input()) {
      inputs_count[input] += 1; // Count inputs.
    }
  }

  for (auto op_index : noop_indices) {
    const auto& op = def_.op(op_index);
    if (op.type() == "GradientGather") {
      if (inputs_count.count(op.output(0)) == 0 &&
          retained_grads.count(op.output(0)) == 0) {
        for (const auto& input : op.input()) {
          inputs_count.erase(input);
        }
      } else {
        string first_input;
        for (const auto& input : op.input()) {
          if (input.empty()) continue;
          if (first_input.empty()) first_input = input;
          splits[input] = {op_index, first_input};
        }
      }
    }
  }

  for (size_t op_index = 0; op_index < def_.op_size(); ++op_index) {
    if (noop_indices.count(op_index)) continue;
    const auto& op = def_.op(op_index);
    optimized_def_.add_op()->CopyFrom(op);
    if (!str::find(op.type(), "Gradient")) continue;
    for (const auto& output : op.output()) {
      const auto& iter = splits.find(output);
      if (iter == splits.end()) continue;
      if (output == iter->second.second) continue;
      auto* gather_op = def_.mutable_op(iter->second.first);
      auto* decouple_op = optimized_def_.add_op();
      decouple_op->CopyFrom(*gather_op);
      if (!gather_op->input().empty()) {
        gather_op->clear_input();
        decouple_op->clear_input();
        decouple_op->add_input(iter->second.second);
      } else {
        decouple_op->add_input(gather_op->output(0));
        const auto& count_iter = inputs_count.find(gather_op->output(0));
        if (count_iter != inputs_count.end()) count_iter->second++;
      }
      decouple_op->add_input(output);
      if (op.arg().empty()) continue;
      const auto& arg = *(op.arg().end() - 1);
      if (arg.name() != "cache_key") continue;
      auto* cache_key = decouple_op->add_arg();
      const auto& option = decouple_op->device_option();
      cache_key->set_name("cache_key");
      cache_key->set_s(
          decouple_op->type() + "/" + str::to(option.device_type()) + ":" +
          str::to(option.device_id()));
    }
  }

  // Initialize buffer pool.
  int buffer_index = 0;
  std::deque<string> buffer_pool;
  auto get_buffer = [&]() mutable {
    if (buffer_pool.empty()) {
      return "BufferGrad_" + str::to(++buffer_index);
    } else {
      auto buffer = buffer_pool.back();
      buffer_pool.pop_back();
      return buffer;
    }
  };

  // Rewrite inputs and outputs.
  for (int op_index = 0; op_index < optimized_def_.op_size(); ++op_index) {
    auto* op = optimized_def_.mutable_op(op_index);
    if (!str::find(op->type(), "Gradient")) continue;
    vector<int> outputs_at(op->output_size(), -1);
    for (int i = 0; i < op->output_size(); ++i) {
      for (int j = 0; j < op->input_size(); ++j) {
        if (op->output(i) != op->input(j)) continue;
        outputs_at[i] = j;
        break;
      }
    }
    // Rewrite inputs.
    vector<string> dead_buffers;
    for (int i = 0; i < op->input_size(); ++i) {
      const string& input = op->input(i);
      const auto& count_iter = inputs_count.find(input);
      if (count_iter == inputs_count.end()) continue;
      count_iter->second--;
      const auto& buffer_iter = outputs_to.find(input);
      if (buffer_iter == outputs_to.end()) continue;
      if (count_iter->second == 0) {
        dead_buffers.emplace_back(buffer_iter->second);
      }
      op->set_input(i, buffer_iter->second);
    }
    // Rewrite outputs.
    for (int i = 0; i < op->output_size(); ++i) {
      const auto& output = op->output(i);
      if (output.empty() || retained_grads.count(output) > 0) continue;
      if (inputs_count.count(output) == 0) {
        op->mutable_output(i)->clear();
        continue;
      }
      if (outputs_at[i] >= 0) {
        op->set_output(i, op->input(outputs_at[i]));
      } else {
        op->set_output(i, outputs_to[output] = get_buffer());
      }
    }
    // Update buffer pool.
    for (auto& buffer : dead_buffers) {
      buffer_pool.emplace_back(buffer);
    }
  }
}

DEFINE_REGISTRY(
    GradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&);

DEFINE_REGISTRY(
    NoGradientRegistry,
    GradientMakerBase,
    const OperatorDef&,
    const vector<string>&);

} // namespace dragon
