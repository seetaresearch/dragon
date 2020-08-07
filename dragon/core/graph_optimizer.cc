#include "dragon/core/graph_optimizer.h"
#include "dragon/core/graph_gradient.h"
#include "dragon/core/operator_schema.h"
#include "dragon/core/workspace.h"

#define GRAPH_TEMPORAL_OUTPUT_MAX_SIZE 2

namespace dragon {

void GraphOptimizer::BuildDAG(const GraphDef& graph) {
  nodes_.clear();
  reference_count_.clear();
  for (int i = 0; i < graph.op_size(); ++i) {
    const auto& op = graph.op(i);
    for (const auto& in : op.input()) {
      reference_count_[in] += 1;
    }
    for (const auto& out : op.output()) {
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

GraphDef GraphOptimizer::EliminateUnused(const GraphDef& graph) {
  // Initialization
  BuildDAG(graph);
  used_.clear();

  // Eliminate the unused nodes
  for (const auto& out : graph.output()) {
    EliminateUnusedNode(out);
  }
  for (const auto& grad_info : graph.grad_info()) {
    const auto grad_y = grad_info.y() + "_grad";
    for (const auto& x : grad_info.xs()) {
      visited_.clear();
      EliminateUnusedNode(grad_y, x + "_grad");
    }
  }

  // Select the used operators
  set<int> selected_op_indices;
  for (auto it : used_) {
    if (nodes_[it.first].op_idx == -1) continue;
    selected_op_indices.insert(nodes_[it.first].op_idx);
  }

  // Prepare the registered placeholders
  Set<string> outputs;
  for (const auto& name : ws_->tensors()) {
    outputs.insert(name);
  }

  // Rewrite graph
  GraphDef graph_v2(graph);
  graph_v2.clear_op();
  for (auto op_idx : selected_op_indices) {
    const auto& op = graph.op(op_idx);
    auto* op_v2 = graph_v2.add_op();
    op_v2->CopyFrom(op);
    // Rewrite inputs
    for (int i = 0; i < op.input_size(); ++i) {
      const auto& in = op.input(i);
      if (!used_[in] || outputs.count(in) == 0) {
        *op_v2->mutable_input(i) = "";
      }
    }
    // Rewrite outputs
    for (int i = 0; i < op.output_size(); ++i) {
      const auto& out = op.output(i);
      if (!used_[out]) {
        *op_v2->mutable_output(i) = "";
      } else {
        outputs.insert(out);
      }
    }
    // Rewrite hand-craft cases
    if (op.type() == "AffineGradient") {
      if (op_v2->output(1).empty()) *op_v2->mutable_input(0) = "";
    } else if (op.type() == "MulGradient") {
      if (op_v2->output(0).empty()) *op_v2->mutable_input(1) = "";
      if (op_v2->output(1).empty()) *op_v2->mutable_input(0) = "";
    } else if (op.type() == "DivGradient") {
      if (op_v2->output(1).empty()) {
        *op_v2->mutable_input(0) = "";
        if (op_v2->output(0).empty()) *op_v2->mutable_input(1) = "";
      }
    }
  }
  return graph_v2;
}

void GraphOptimizer::PlanInplace(
    const GraphDef& graph,
    Map<string, Set<string>>& output_aliases) {
  // Initialization
  BuildDAG(graph);

  // Generate aliases map to apply in-place
  for (const auto& iter : reference_count_) {
    const auto& in = iter.first;
    if (iter.second == 1 && !in.empty() && nodes_[in].childs.size() > 0) {
      const auto& op = nodes_[nodes_[in].childs[0]].op_def;
      const auto* schema = OpSchemaRegistry::Schema(op.type());
      for (int i = 0; i < op.input_size(); ++i) {
        if (op.input(i) == in) {
          for (int j = 0; j < op.output_size(); ++j) {
            if (schema->CheckInplace(i, j)) {
              output_aliases[op.output(j)].insert(in);
            }
          }
        }
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

  // Check the mirror stage setting
  for (const auto& op : graph.op()) {
    if (str::find(op.type(), "Gradient")) continue;
    bool mirror_stage = false;
    for (auto& arg : op.arg()) {
      if (arg.name() == "mirror_stage") {
        mirror_stage |= (bool)arg.i();
      }
    }
    if (mirror_stage) {
      // We only assume X(0) can be recomputed
      rename_map[op.input(0)] = "placeholder";
    }
  }

  // Allocate the temporal buffers
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
          v2_name = "/share/buffer/symbol:" + str::to(j);
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

GraphDef GraphOptimizer::SimulateGC(const GraphDef& graph) {
  Set<string> blacklist = {""};
  Map<string, int> ref_count;
  Map<string, string> rename_map;
  static Set<string> star_ops = {"Shape"};

  // Prepare the pool
  int buffer_idx = 0;
  std::deque<string> pool;
  auto get_buffer = [&]() mutable {
    if (pool.empty()) {
      return "/share/buffer/output:" + str::to(buffer_idx++);
    } else {
      auto buffer = pool.back();
      pool.pop_back();
      return buffer;
    }
  };

  // Count the references
  for (const auto& op : graph.op()) {
    for (const auto& in : op.input()) {
      ref_count[in] += 1;
    }
  }

  // Preserve the graph outputs
  for (auto& out : graph.output()) {
    blacklist.insert(out);
  }

  // Rewrite the inputs and outputs
  auto graph_v2(graph);
  for (int op_idx = 0; op_idx < graph.op_size(); ++op_idx) {
    const auto& op = graph.op(op_idx);
    auto* op_v2 = graph_v2.mutable_op(op_idx);
    // Ignore the init ops
    if (op.input_size() == 0) continue;
    // We need to collect the dead buffers.
    // Reuse them when current operator is done.
    vector<string> dead_buffers;
    // Rewrite inputs
    for (int i = 0; i < op.input_size(); ++i) {
      const auto& name = op.input(i);
      if (rename_map.count(name)) {
        *op_v2->mutable_input(i) = rename_map[name];
      }
      ref_count[name]--;
      if (ref_count[name] == 0 &&
          str::startswith(op_v2->input(i), "/share/buffer/output:")) {
        dead_buffers.push_back(op_v2->input(i));
      }
    }
    // Rewrite outputs
    if (!star_ops.count(op.type())) {
      for (int i = 0; i < op.output_size(); ++i) {
        const auto& name = op.output(i);
        bool inplace_flag = false;
        if (blacklist.count(name)) continue;
        for (const auto& input : op.input())
          if (name == input) inplace_flag = true;
        if (inplace_flag) {
          *op_v2->mutable_output(i) = op_v2->input(i);
        } else {
          rename_map[name] = *op_v2->mutable_output(i) = get_buffer();
        }
      }
    }
    // Update the pool
    for (auto& buffer : dead_buffers) {
      pool.emplace_back(buffer);
    }
  }
  return graph_v2;
}

void GraphOptimizer::EliminateUnusedNode(
    const string& source,
    const string& sink) {
  if (visited_.count(source)) return;
  visited_[source] = false;
  for (const auto& next : nodes_[source].childs) {
    if (next == sink) {
      visited_[next] = used_[next] = true;
      visited_[source] = used_[source] = true;
      return;
    }
    EliminateUnusedNode(next, sink);
    if (visited_[next]) {
      visited_[source] = used_[source] = true;
    }
  }
}

void GraphOptimizer::EliminateUnusedNode(const string& sink) {
  std::queue<const string*> q;
  q.push(&sink);
  while (!q.empty()) {
    const auto& source = *q.front();
    q.pop();
    used_[source] = true;
    for (const auto& last : nodes_[source].parents) {
      if (used_.count(last)) continue;
      q.push(&last);
    }
  }
}

} // namespace dragon
