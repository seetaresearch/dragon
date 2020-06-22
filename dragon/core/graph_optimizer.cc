#include "dragon/core/graph_optimizer.h"
#include "dragon/core/graph_gradient.h"
#include "dragon/core/operator_schema.h"
#include "dragon/core/workspace.h"

#define GRAPH_TEMPORAL_OUTPUT_MAX_SIZE 2

namespace dragon {

void GraphOptimizer::BuildDAG(const GraphDef& input_def) {
  dag_.clear();
  colored_.clear();
  reference_count_.clear();
  for (int i = 0; i < input_def.op_size(); ++i) {
    const auto& op = input_def.op(i);
    for (const auto& u : op.input()) {
      reference_count_[u] += 1;
    }
    for (const auto& v : op.output()) {
      vector<string> u_set(op.input().begin(), op.input().end());
      if (u_set.empty()) u_set.resize(op.output_size());
      for (const auto& u : u_set) {
        dag_[v].parents.push_back(u);
        dag_[u].childs.push_back(v);
        dag_[v].op_idx = i;
      }
      dag_[v].op_def = op;
    }
  }
}

GraphDef GraphOptimizer::PruneNodes(const GraphDef& input_def) {
  // Initialization
  BuildDAG(input_def);

  // Backward pass from targets
  for (const auto& target : input_def.output()) {
    if (colored_[target]) continue;
    BackwardPrunePass(target);
  }

  // Forward pass from gradients
  for (const auto& gradient : input_def.gradient()) {
    auto u = gradient.cost() + "_grad";
    auto v = gradient.wrt() + "_grad";
    if (ws_->HasTensor(u)) u = ws_->GetTensor(u)->name();
    if (ws_->HasTensor(v)) v = ws_->GetTensor(v)->name();
    visited_.clear();
    ForwardPrunePass(u, v, vector<string>({u}));
  }

  // Select all colored operators
  set<int> selected_op_indices;
  for (auto it : colored_) {
    if (dag_[it.first].op_idx == -1) continue;
    selected_op_indices.insert(dag_[it.first].op_idx);
  }

  // Remove the tensors that can not be produced
  Set<string> outputs;
  for (const auto& name : ws_->tensors()) {
    outputs.insert(name);
  }

  // Generate the final op sequence
  map<int, OperatorDef> final_sequence;

  for (auto op_idx : selected_op_indices) {
    const auto& op = input_def.op(op_idx);
    auto new_op(input_def.op(op_idx));
    // Rewrite inputs
    for (int i = 0; i < op.input_size(); ++i) {
      const auto& input = op.input(i);
      if (!colored_[input] || outputs.count(input) == 0)
        *new_op.mutable_input(i) = "";
    }
    // Rewrite outputs
    for (int i = 0; i < op.output_size(); ++i) {
      const auto& output = op.output(i);
      if (!colored_[output]) {
        *new_op.mutable_output(i) = "";
      } else {
        outputs.insert(output);
      }
    }
    // Rewrite hand-craft cases
    if (op.type() == "AffineGradient") {
      if (new_op.output(1).empty()) *new_op.mutable_input(0) = "";
    } else if (op.type() == "MulGradient") {
      if (new_op.output(0).empty()) *new_op.mutable_input(1) = "";
      if (new_op.output(1).empty()) *new_op.mutable_input(0) = "";
    } else if (op.type() == "DivGradient") {
      if (new_op.output(1).empty()) {
        *new_op.mutable_input(0) = "";
        if (new_op.output(0).empty()) *new_op.mutable_input(1) = "";
      }
    }
    // Push into the final sequence
    final_sequence[op_idx].CopyFrom(new_op);
  }

  // Done!
  GraphDef output_def(input_def);
  output_def.clear_op();
  for (auto it : final_sequence)
    output_def.add_op()->CopyFrom(it.second);
  return output_def;
}

void GraphOptimizer::AddInplace(
    const GraphDef& input_def,
    Map<string, Set<string>>& output_aliases) {
  // Initialization
  BuildDAG(input_def);

  // Generate runtime aliases map
  for (auto& u_iter : reference_count_) {
    if (u_iter.second == 1 && !u_iter.first.empty() &&
        dag_[u_iter.first].childs.size() > 0) {
      const auto& u = u_iter.first;
      const auto& v0 = dag_[u].childs[0];
      const auto& op_def = dag_[v0].op_def;
      const auto* op_schema = OpSchemaRegistry::Schema(op_def.type());
      for (int i = 0; i < op_def.input_size(); ++i)
        for (int j = 0; j < op_def.output_size(); ++j)
          if (op_schema->CheckInplace != nullptr && op_def.input(i) == u &&
              op_schema->CheckInplace(i, j))
            output_aliases[op_def.output(j)].insert(u);
    }
  }
}

GraphDef GraphOptimizer::MirrorStage(
    const GraphDef& input_def,
    Map<string, vec32_t>& op_indices) {
  GraphDef output_def(input_def);
  Map<string, set<int>> fake_op_indices;
  Map<string, string> rename_map;
  Map<string, int> versions;

  // Check mirror stage
  for (const auto& op : input_def.op()) {
    if (str::find(op.type(), "Gradient")) continue;
    bool mirror_stage = false;
    for (auto& arg : op.arg())
      if (arg.name() == "mirror_stage") mirror_stage |= (bool)arg.i();
    if (mirror_stage) {
      // We only assume X(0) can be recomputed
      rename_map[op.input(0)] = "placeholder";
    }
  }

  // Allocate the temporal buffers
  string v2_name, version_name;
  for (int op_idx = 0; op_idx < input_def.op_size(); ++op_idx) {
    const auto& op = input_def.op(op_idx);
    auto* new_op = output_def.mutable_op(op_idx);
    vector<string> used_buffers;
    for (int i = 0; i < op.input_size(); ++i) {
      const auto& it = rename_map.find(op.input(i));
      if (it != rename_map.end() && it->second != "placeholder") {
        *new_op->mutable_input(i) = it->second;
        used_buffers.emplace_back(it->second);
      }
    }
    for (int i = 0; i < op.output_size(); ++i) {
      bool inplace_flag = false;
      for (const auto& u : op.input())
        if (u == op.output(i)) inplace_flag = true;
      if (rename_map.count(op.output(i))) {
        if (inplace_flag && rename_map[op.output(i)] != "placeholder") {
          *new_op->mutable_output(i) = rename_map[op.output(i)];
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
        *new_op->mutable_output(i) = rename_map[op.output(i)] =
            v2_name + version_name;
      }
    }
  }

  // Plan the minimum recomputing ops for temporal buffers
  for (int i = 0; i < input_def.op_size(); ++i) {
    const auto& input_op = input_def.op(i);
    const auto& output_op = output_def.op(i);

    /*
     * DP(v) = {DP(u) if input(u) != output(u) else {}} + {i}
     */

    set<int> minimum_ops = {i};
    for (int j = 0; j < input_op.input_size(); ++j) {
      if (input_op.input(j) != output_op.input(j)) {
        for (auto idx : fake_op_indices[input_op.input(j)])
          minimum_ops.insert(idx);
      }
    }
    for (const auto& output : input_op.output()) {
      for (auto idx : minimum_ops)
        fake_op_indices[output].insert(idx);
    }
  }

  // Bind to the renamed tensors
  for (const auto& it : rename_map) {
    for (auto op_idx : fake_op_indices[it.first])
      op_indices[it.second].push_back(op_idx);
  }

  // Done!
  return output_def;
}

GraphDef GraphOptimizer::SimulateGC(const GraphDef& input_def) {
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
  for (const auto& op : input_def.op()) {
    for (const auto& input : op.input())
      ref_count[input] += 1;
  }

  // We should preserve the targets
  for (auto& e : input_def.output()) {
    blacklist.insert(e);
  }

  // Rewritten the inputs and outputs
  auto output_def(input_def);
  for (int op_idx = 0; op_idx < input_def.op_size(); ++op_idx) {
    const auto& op = input_def.op(op_idx);
    auto* new_op = output_def.mutable_op(op_idx);

    // Ignore the init ops
    if (op.input_size() == 0) continue;

    // We need to collect the dead buffers
    // Reuse them when current operator is done
    vector<string> dead_buffers;

    // Rewrite inputs
    for (int i = 0; i < op.input_size(); ++i) {
      const auto& name = op.input(i);
      if (rename_map.count(name)) {
        *new_op->mutable_input(i) = rename_map[name];
      }
      ref_count[name]--;
      if (ref_count[name] == 0 &&
          str::startswith(new_op->input(i), "/share/buffer/output:")) {
        dead_buffers.push_back(new_op->input(i));
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
          *new_op->mutable_output(i) = new_op->input(i);
        } else {
          rename_map[name] = *new_op->mutable_output(i) = get_buffer();
        }
      }
    }

    // Update the pool
    for (auto& buffer : dead_buffers) {
      pool.emplace_back(buffer);
    }
  }

  return output_def;
}

void GraphOptimizer::ForwardPrunePass(
    const string& u,
    const string& leaf,
    const vector<string>& path) {
  if (visited_.count(u)) {
    if (visited_[u])
      for (const auto& node : path)
        visited_[node] = colored_[node] = true;
    return;
  }
  visited_[u] = false;
  for (int i = 0; i < dag_[u].childs.size(); ++i) {
    auto v = dag_[u].childs[i];
    auto new_path(path);
    new_path.push_back(v);
    if (v == leaf) {
      for (const auto& node : new_path)
        visited_[node] = colored_[node] = true;
      return;
    }
    ForwardPrunePass(v, leaf, new_path);
  }
}

void GraphOptimizer::BackwardPrunePass(const string& v) {
  colored_[v] = true;
  for (int i = 0; i < dag_[v].parents.size(); ++i) {
    auto u = dag_[v].parents[i];
    if (colored_.count(u)) continue;
    BackwardPrunePass(u);
  }
}

} // namespace dragon
