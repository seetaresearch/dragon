#include "dragon/core/graph_optimizer.h"
#include "dragon/core/operator_schema.h"
#include "dragon/core/workspace.h"

#define GRAPH_TEMPORAL_OUTPUT_MAX_SIZE 2

namespace dragon {

void GraphOptimizer::BuildDAG(const GraphDef& graph) {
  nodes_.clear();
  inputs_count_.clear();
  for (int i = 0; i < graph.op_size(); ++i) {
    const auto& op = graph.op(i);
    for (const auto& in : op.input()) {
      inputs_count_[in] += 1;
    }
    for (const auto& out : op.output()) {
      if (out.empty()) continue;
      if (op.input().empty()) {
        nodes_[""].childs.push_back(out);
        nodes_[out].parents.push_back("");
      } else {
        for (const auto& in : op.input()) {
          nodes_[in].childs.push_back(out);
          nodes_[out].parents.push_back(in);
        }
      }
      nodes_[out].op_idx = i;
      nodes_[out].op_def = op;
    }
  }
}

void GraphOptimizer::PlanInplace(
    const GraphDef& graph,
    Map<string, Set<string>>& output_aliases) {
  // Initialization.
  BuildDAG(graph);
  // Generate aliases map to apply in-place.
  for (const auto& iter : inputs_count_) {
    if (iter.second > 1 || iter.first.empty()) continue;
    const auto& input = iter.first;
    const auto& input_node = nodes_[input];
    if (input_node.childs.empty() || input_node.parents.empty()) continue;
    const auto& op = nodes_[input_node.childs[0]].op_def;
    const auto* schema = OpSchemaRegistry::Schema(op.type());
    for (int i = 0; i < op.input_size(); ++i) {
      if (op.input(i) != input) continue;
      for (int j = 0; j < op.output_size(); ++j) {
        if (!schema->CheckInplace(i, j)) continue;
        output_aliases[op.output(j)].insert(input);
      }
    }
  }
}

GraphDef GraphOptimizer::PlanCheckpoint(
    const GraphDef& graph,
    Map<string, vec32_t>& subgraph_indices) {
  GraphDef graph_v2(graph);
  Map<string, set<int>> op_indices;
  Map<string, string> rename_map;
  Map<string, int> versions;

  // Check the mirror stage setting.
  for (const auto& op : graph.op()) {
    if (str::find(op.type(), "Gradient")) continue;
    bool mirror_stage = false;
    for (auto& arg : op.arg()) {
      if (arg.name() == "mirror_stage") {
        mirror_stage |= (bool)arg.i();
      }
    }
    if (mirror_stage) {
      // We only assume X(0) can be recomputed.
      rename_map[op.input(0)] = "placeholder";
    }
  }

  // Allocate the temporal buffers.
  string v2_name, version_name;
  for (int op_idx = 0; op_idx < graph.op_size(); ++op_idx) {
    const auto& op = graph.op(op_idx);
    auto* op_v2 = graph_v2.mutable_op(op_idx);
    vector<string> used_buffers;
    for (int i = 0; i < op.input_size(); ++i) {
      const auto& it = rename_map.find(op.input(i));
      if (it != rename_map.end() && it->second != "placeholder") {
        *op_v2->mutable_input(i) = it->second;
        used_buffers.emplace_back(it->second);
      }
    }
    for (int i = 0; i < op.output_size(); ++i) {
      bool inplace_flag = false;
      for (const auto& in : op.input()) {
        if (in == op.output(i)) inplace_flag = true;
      }
      if (rename_map.count(op.output(i))) {
        if (inplace_flag && rename_map[op.output(i)] != "placeholder") {
          *op_v2->mutable_output(i) = rename_map[op.output(i)];
          continue;
        }
        for (int j = 0; j < GRAPH_TEMPORAL_OUTPUT_MAX_SIZE; ++j) {
          v2_name = "shared/buffer/output:" + str::to(j);
          for (const auto& buffer : used_buffers)
            if (str::find(buffer, v2_name)) {
              v2_name.clear();
            }
          if (!v2_name.empty()) {
            used_buffers.emplace_back(v2_name);
            break;
          }
        }
        CHECK(!v2_name.empty()) << "\nNo enough buffers for outputs.";
        ws_->CreateTensor(v2_name)->set_version(0);
        version_name = "/ver:" + str::to(versions[v2_name]++);
        *op_v2->mutable_output(i) = rename_map[op.output(i)] =
            v2_name + version_name;
      }
    }
  }

  // Determine the recomputing ops for temporal buffers
  for (int i = 0; i < graph.op_size(); ++i) {
    const auto &op = graph.op(i), &op_v2 = graph_v2.op(i);
    set<int> recomputing_ops = {i};
    for (int j = 0; j < op.input_size(); ++j) {
      if (op.input(j) != op_v2.input(j)) {
        for (auto op_idx : op_indices[op.input(j)]) {
          recomputing_ops.insert(op_idx);
        }
      }
    }
    for (const auto& out : op.output()) {
      for (auto op_idx : recomputing_ops) {
        op_indices[out].insert(op_idx);
      }
    }
  }

  // Bind to the renamed tensors
  for (const auto& it : rename_map) {
    for (auto op_idx : op_indices[it.first]) {
      subgraph_indices[it.second].push_back(op_idx);
    }
  }

  // Done
  return graph_v2;
}

GraphDef GraphOptimizer::EliminateIntermediates(const GraphDef& graph) {
  Set<string> required_outputs;
  Map<string, int> inputs_count;
  Map<string, string> outputs_to_buffers;
  static Set<string> skip_ops = {"Shape"};

  // Prepare pool.
  int buffer_idx = 0;
  std::deque<string> pool;
  auto get_buffer = [&]() mutable {
    if (pool.empty()) {
      return "shared/buffer/output:" + str::to(++buffer_idx);
    } else {
      auto buffer = pool.back();
      pool.pop_back();
      return buffer;
    }
  };

  // Count inputs.
  for (const auto& op : graph.op()) {
    for (const auto& input : op.input()) {
      inputs_count[input] += 1;
    }
  }

  // Initialize the required outputs before optimization.
  for (const auto& output : graph.output()) {
    required_outputs.insert(output);
  }

  // Rewrite the inputs and outputs.
  auto graph_v2(graph);
  for (int op_idx = 0; op_idx < graph.op_size(); ++op_idx) {
    const auto& op = graph.op(op_idx);
    if (op.input_size() == 0) continue;
    auto* op_v2 = graph_v2.mutable_op(op_idx);
    // Check output aliases.
    vec32_t output_aliases(op.output_size(), -1);
    for (int i = 0; i < op.output_size(); ++i) {
      for (int j = 0; j < op.input_size(); ++j) {
        if (op.output(i) != op.input(j)) continue;
        output_aliases[i] = j;
        break;
      }
    }
    // Rewrite inputs.
    vector<string> dead_buffers;
    for (int i = 0; i < op.input_size(); ++i) {
      const auto& input = op.input(i);
      const auto& count_iter = inputs_count.find(input);
      count_iter->second--;
      const auto& buffer_iter = outputs_to_buffers.find(input);
      if (buffer_iter == outputs_to_buffers.end()) continue;
      if (count_iter->second == 0) {
        dead_buffers.emplace_back(buffer_iter->second);
      }
      op_v2->set_input(i, buffer_iter->second);
    }
    if (skip_ops.count(op.type())) continue;
    // Rewrite outputs.
    for (int i = 0; i < op.output_size(); ++i) {
      const auto& output = op.output(i);
      if (output.empty() || required_outputs.count(output) > 0) continue;
      if (output_aliases[i] >= 0) {
        op_v2->set_output(i, op_v2->input(output_aliases[i]));
      } else {
        *op_v2->mutable_output(i) = outputs_to_buffers[output] = get_buffer();
      }
    }
    // Update pool.
    for (auto& buffer : dead_buffers) {
      pool.emplace_back(buffer);
    }
  }
  return graph_v2;
}

} // namespace dragon
