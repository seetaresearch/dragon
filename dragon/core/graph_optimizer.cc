#include "dragon/core/graph_optimizer.h"
#include "dragon/core/operator_schema.h"
#include "dragon/core/workspace.h"

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
    Map<string, Set<string>>& sources) {
  // Initialization.
  BuildDAG(graph);
  // Add source for outputs to apply in-place.
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
        sources[op.output(j)].insert(input);
      }
    }
  }
}

GraphDef GraphOptimizer::EliminateIntermediates(const GraphDef& graph) {
  Set<string> graph_outputs;
  Map<string, int> inputs_count;
  Map<string, string> outputs_to;
  static Set<string> noop_types = {"Shape"};

  auto optimized_graph(graph);

  // Initialize buffer pool.
  int buffer_idx = 0;
  std::deque<string> buffer_pool;
  auto get_buffer = [&]() mutable {
    if (buffer_pool.empty()) {
      return "Buffer_" + str::to(++buffer_idx);
    } else {
      auto buffer = buffer_pool.back();
      buffer_pool.pop_back();
      return buffer;
    }
  };

  // Count inputs.
  for (const auto& op : graph.op()) {
    for (const auto& input : op.input()) {
      inputs_count[input] += 1;
    }
  }

  // Initialize graph outputs before optimization.
  for (const auto& output : graph.output()) {
    graph_outputs.insert(output);
  }

  // Rewrite inputs and outputs.
  for (int op_idx = 0; op_idx < graph.op_size(); ++op_idx) {
    auto* op = optimized_graph.mutable_op(op_idx);
    if (op->input_size() == 0) continue;
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
      const auto& input = op->input(i);
      const auto& count_iter = inputs_count.find(input);
      count_iter->second--;
      const auto& buffer_iter = outputs_to.find(input);
      if (buffer_iter == outputs_to.end()) continue;
      if (count_iter->second == 0) {
        dead_buffers.emplace_back(buffer_iter->second);
      }
      op->set_input(i, buffer_iter->second);
    }
    if (noop_types.count(op->type())) continue;
    // Rewrite outputs.
    for (int i = 0; i < op->output_size(); ++i) {
      const auto& output = op->output(i);
      if (output.empty() || graph_outputs.count(output) > 0) continue;
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

  return optimized_graph;
}

} // namespace dragon
