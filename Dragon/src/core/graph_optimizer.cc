#include "core/workspace.h"
#include "core/graph_gradient.h"
#include "core/operator_schema.h"
#include "core/graph_optimizer.h"

#define GRAPH_TEPORAL_OUTPUT_MAX_SIZE 2

namespace dragon {

/*! Prune the redundant nodes (-O1) */

GraphDef GraphOptimizer::PruneNodes(const GraphDef& input_def) {
    dag_.clear(); colored_.clear();
    // Build DAG
    for (int i = 0; i < input_def.op_size(); ++i) {
        const OperatorDef& op = input_def.op(i);
        for (const auto& v : op.output()) {
            vector<string> sp_u;
            if (!op.input_size()) sp_u.resize(op.output_size(), "");
            else sp_u.assign(op.input().begin(), op.input().end());
            for (const auto& u : sp_u) {
                if (u == "ignore") continue;
                dag_[v].parents.push_back(u);
                dag_[u].childs.push_back(v);
                dag_[v].op_idx = i;
            }
            dag_[v].op_def = op;
        }
    }

    // Backward dyeing for all solving targets
    for (const auto& target : input_def.output()) {
        if (colored_[target]) continue;
        BackwardPruneTraversal(target);
    }

    // Forward dyeing through connected path for all gradients
    for (const auto& gradient : input_def.gradient()) {
        string u = gradient.cost() + "_grad";
        string v = gradient.wrt() + "_grad";
        if (ws_->HasTensor(u)) u = ws_->GetTensor(u)->name();
        if (ws_->HasTensor(v)) v = ws_->GetTensor(v)->name();
        visited_.clear();
        ForwardPruneTraversal(u, v, vector<string>({ u }));
    }

    // Select all colored operators
    // Note that we use set to keep topo-order
    set<int> selected_op_indices;
    for (auto it : colored_) {
        if (dag_[it.first].op_idx == -1) continue;
        selected_op_indices.insert(dag_[it.first].op_idx);
    }

    // Remove the tensors that can not be produced(redundant)
    Set<string> outputs;
    // Check if having feeded tensors
    for (const auto& e : ws_->GetTensors()) outputs.insert(e);
    // Note that we use map to keep topo-order
    map<int, OperatorDef> final_sequence;

    for (auto it : selected_op_indices) {
        OperatorDef op_def;
        op_def.CopyFrom(input_def.op(it));
        // Rewritten for inputs
        for (int i = 0; i < input_def.op(it).input_size(); ++i) {
            string input = input_def.op(it).input(i);
            if (!colored_[input] || !outputs.count(input))
                *op_def.mutable_input(i) = "ignore";
        }
        // Rewritten for outputs
        for (int i = 0; i < input_def.op(it).output_size(); ++i) {
            string output = input_def.op(it).output(i);
            if (!colored_[output]) *op_def.mutable_output(i) = "ignore";
            else outputs.insert(op_def.output(i));
        }
        // Rewritten for some hand-craft cases
        if (op_def.type() == "AffineGradient") {
            // Trigger in-place if not solving dAlpha
            if (op_def.output(1) == "ignore")
                *op_def.mutable_input(0) = "ignore";
        } else if (op_def.type() == "MulGradient" ||
                   op_def.type() == "RMulGradient") {
            if (op_def.output(0) == "ignore")
                *op_def.mutable_input(1) = "ignore";
            if (op_def.output(1) == "ignore")
                *op_def.mutable_input(0) = "ignore";
        } else if (op_def.type() == "DivGradient" ||
                   op_def.type() == "RDivGradient") {
            // dX2 requires both X1 and X2
            if (op_def.output(1) == "ignore") {
                *op_def.mutable_input(0) = "ignore";
                if (op_def.output(0) == "ignore")
                    *op_def.mutable_input(1) = "ignore";
            }
        }
        // Push into the final sequence
        final_sequence[it].CopyFrom(op_def);
    }

    // Done!
    GraphDef output_def(input_def); output_def.clear_op();
    for (auto it : final_sequence)
        output_def.add_op()->CopyFrom(it.second);
    return output_def;
}

/*! Add the inplace for outputs (-O2) */

GraphDef GraphOptimizer::AddInplace(const GraphDef& input_def) {
    dag_.clear(); renamed_.clear();
    // Build DAG
    for (int i = 0; i < input_def.op_size(); ++i) {
        const OperatorDef& op = input_def.op(i);
        for (const auto& v : op.output()) {
            vector<string> sp_u;
            if (!op.input_size()) sp_u.resize(op.output_size(), "");
            else sp_u.assign(op.input().begin(), op.input().end());
            for (const auto& u : sp_u) {
                if (u == "ignore") continue;
                dag_[v].parents.push_back(u);
                dag_[u].childs.push_back(v);
                dag_[v].op_idx = i;
            }
            dag_[v].op_def = op;
        }
    }

    // Forward dyeing to search available tensors that be shared
    for (const auto& op : input_def.op()) {
        for (const auto& u : op.input()) InplaceTraversal(u, u);
        for (const auto& v : op.output()) InplaceTraversal(v, v);
    }

    GraphDef output_def(input_def);

    // We need a whitelist to persist inputs and outputs
    // Their content should not be overwritten
    Set<string> whitelist;
    for (const auto& e : output_def.input()) whitelist.insert(e);
    for (const auto& e : output_def.output()) whitelist.insert(e);

    // Rename to create in-place
    for (int i = 0; i < input_def.op_size(); ++i) {
        const OperatorDef& op = input_def.op(i);
        for (int j = 0; j < op.input_size(); ++j) {
            if (whitelist.count(op.input(j)) == 0 &&
                renamed_.count(op.input(j)) &&
                ws_->SetTensorAlias(renamed_[op.input(j)], op.input(j)))
                    *output_def.mutable_op(i)->mutable_input(j)
                        = renamed_[op.input(j)];
        }
        // Handle handcraft cases
        if (op.type() == "BiasAddGradient") {
            // Share dY for dX at BiasAddGradientOp
            renamed_[op.output(0)] = output_def.op(i).input(1);
        }
        for (int j = 0; j < op.output_size(); ++j) {
            if (whitelist.count(op.output(j)) == 0 &&
                renamed_.count(op.output(j)) &&
                ws_->SetTensorAlias(renamed_[op.output(j)], op.output(j)))
                    *output_def.mutable_op(i)->mutable_output(j)
                        = renamed_[op.output(j)];
        }
    }

    // Done!
    return output_def;
}

/*! Plan the recomputing for inputs (-O3) */

GraphDef GraphOptimizer::MirrorStage(
    const GraphDef&                  input_def,
    Map<string, vector<int> >&       op_indices) {
    GraphDef output_def(input_def);
    Map<string, set<int> > fake_op_indices;
    Map<string, string> rename_map;
    Map<string, int> versions;

    // Check mirror stage
    for (const auto& op : input_def.op()) {
        if (op.type().find("Gradient") != string::npos) continue;
        bool mirror_stage = false;
        for (auto& arg : op.arg())
            if (arg.name() == "mirror_stage")
                mirror_stage |= (bool)arg.i();
        if (mirror_stage) {
            // We only assume Input(0) can be recomputed
            rename_map[op.input(0)] = "placeholder";
        }
    }

    // Allocate the temporal buffers
    string v2_name, version_name;
    for (int i = 0; i < input_def.op_size(); ++i) {
        const OperatorDef& op = input_def.op(i);
        OperatorDef* op_v2 = output_def.mutable_op(i);
        vector<string> used_buffers;
        for (int j = 0; j < op.input_size(); ++j) {
            const auto& it = rename_map.find(op.input(j));
            if (it != rename_map.end() &&
                it->second != "placeholder") {
                *op_v2->mutable_input(j) = it->second;
                used_buffers.emplace_back(it->second);
            }
        }
        for (int j = 0; j < op.output_size(); ++j) {
            bool inplace_flag = false;
            for (const auto& u : op.input())
                if (u == op.output(j)) inplace_flag = true;
            if (rename_map.count(op.output(j))) {
                if (inplace_flag && rename_map[op.output(j)]
                            != "placeholder")  {
                    *op_v2->mutable_output(j) =
                        rename_map[op.output(j)];
                    continue;
                }
                for (int k = 0; k < GRAPH_TEPORAL_OUTPUT_MAX_SIZE; ++k) {
                    v2_name = "/share/buffer/symbol:" + std::to_string(k);
                    for (const auto& buffer : used_buffers)
                        if (buffer.find(v2_name) != string::npos) { v2_name.clear(); }
                    if (!v2_name.empty()) { used_buffers.emplace_back(v2_name); break; }
                }
                CHECK(!v2_name.empty()) << "\nNo enough buffers for outputs.";
                ws_->CreateTensor(v2_name)->set_version(0);
                if (!versions.count(v2_name)) versions[v2_name] = 0;
                version_name = "/ver:" + std::to_string(versions[v2_name]++);
                *op_v2->mutable_output(j) = rename_map[op.output(j)] =
                              v2_name + version_name;
            }
        }
    }

    // Plan the minimum recomputing ops for temporal buffers
    for (int i = 0; i < input_def.op_size(); ++i) {
        const OperatorDef& input_op = input_def.op(i);
        const OperatorDef& output_op = output_def.op(i);

        /* ----------------------------------------------------------
         *
         *                   Optimal Substructure
         *
         *    DP(v) = {DP(u) if input(u) != output(u) else {}} + {i}
         *
         * ---------------------------------------------------------- */

        set<int> minimum_ops = {i};
        for (int j = 0; j < input_op.input_size(); ++j) {
            if (input_op.input(j) != output_op.input(j)) {
                if (!fake_op_indices.count(input_op.input(j)))
                    fake_op_indices[input_op.input(j)] = set<int>();
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
        op_indices[it.second] = vector<int>();
        for (auto op_idx : fake_op_indices[it.first])
            op_indices[it.second].push_back(op_idx);
    }

    // Done!
    return output_def;
}

/*! Allocate the buffer for outputs (-O3) */

GraphDef GraphOptimizer::SimulateGC(const GraphDef& input_def) {
    GraphDef output_def(input_def);
    Set<string> targets;
    Map<string, int> ref_count;
    Map<string, string> rename_map;

    Set<string> dimension_ops = {
        "Shape", "Reshape", "Flatten",
            "ExpandDims", "Squeeze",
    }, blacklist_outputs, star_ops = {
        "Reshape", "Crop", "NNResize",
            "BilinearResize",
    };

    // Prepare the Outputs Pool
    int temporary_idx = 0;
    std::deque<string> outputs_pool;
    auto get_temporary_output = [&]() mutable {
        if (outputs_pool.empty()) {
            return "/share/buffer/output:" +
                std::to_string(temporary_idx++);
        } else {
            string temporary_output = outputs_pool.back();
            outputs_pool.pop_back();
            return temporary_output;
        }
    };

    // Count the references
    for (auto& op : input_def.op()) {
        for (auto& input : op.input())
            ref_count[input] += 1;
        if (dimension_ops.count(op.type()))
            blacklist_outputs.insert(op.input(0));
        if (star_ops.count(op.type())) {
            for (const auto& arg : op.arg())
                if (arg.name() == "shape_like")
                    blacklist_outputs.insert(arg.s());
        }
    }

    // We should preserve the targets
    for (auto& e : input_def.output()) targets.insert(e);

    // Rewritten the inputs and outputs
    string ori_name; bool inplace_flag;
    for (int i = 0; i < input_def.op_size(); ++i) {
        vector<string> GC;
        const OperatorDef& op = input_def.op(i);
        OperatorDef* op_v2 = output_def.mutable_op(i);

        // Ignore the init operators
        if (op.input_size() == 0) continue;

        // Analyze the inputs
        for (int j = 0; j < op.input_size(); ++j) {
            ori_name = op.input(j);
            if (rename_map.count(ori_name)) {
                *op_v2->mutable_input(j) =
                    rename_map[ori_name];
            }
            ref_count[ori_name]--;
            if (ref_count[ori_name] == 0 && op_v2->input(j).find(
                "/share/buffer/output:") != string::npos)
                    GC.push_back(op_v2->input(j));
        }

        // Allocate the buffers
        if (!dimension_ops.count(op.type())) {
            for (int j = 0; j < op.output_size(); ++j) {
                inplace_flag = false;
                ori_name = op.output(j);
                if (targets.count(ori_name) ||
                        blacklist_outputs.count(ori_name))
                            continue;
                for (const auto& input : op.input())
                    if (ori_name == input) inplace_flag = true; 
                if (inplace_flag) {
                    *op_v2->mutable_output(j) = op_v2->input(j);
                } else {
                    rename_map[ori_name] =
                        *op_v2->mutable_output(j) =
                            get_temporary_output();
                }
            }
        }

        // Update the pool from GC
        for (auto& e : GC) outputs_pool.emplace_back(e);
    }

    return output_def;
}

/*! Traverse from input gradients to dying the nodes */

void GraphOptimizer::ForwardPruneTraversal(
    const string&               u,
    const string&               leaf,
    const vector<string>&       path) {
    if (visited_.count(u)) {
        if (visited_[u])
            for (const auto& node : path)
                visited_[node] = colored_[node] = true;
        return;
    }
    visited_[u] = false;
    for (int i = 0; i < dag_[u].childs.size(); ++i) {
        string v = dag_[u].childs[i];
        vector<string> new_path(path);
        new_path.push_back(v);
        if (v == leaf) {
            for (const auto& node : new_path)
                visited_[node] = colored_[node] = true;
            return;
        }
        ForwardPruneTraversal(v, leaf, new_path);
    }
}

/*! Traverse from targets to dying the nodes */

void GraphOptimizer::BackwardPruneTraversal(const string& v) {
    colored_[v] = true;
    for (int i = 0; i < dag_[v].parents.size(); ++i) {
        string u = dag_[v].parents[i];
        if (colored_.count(u)) continue;
        BackwardPruneTraversal(u);
    }
}

/*! Traverse from inputs to find the available inplace chain */

void GraphOptimizer::InplaceTraversal(
    const string&               u,
    const string&               ancestor) {
    if (renamed_.count(u)) return;
    renamed_[u] = ancestor;
    if (dag_[u].childs.size() == 1) {
        auto& v = dag_[u].childs[0];
        auto& op_def = dag_[v].op_def;
        auto* schema = OpSchemaRegistry::Schema(op_def.type());
        if (schema->AllowInplace())
            for (int i = 0; i < op_def.input_size(); ++i)
                if (op_def.input(i) == u &&
                    schema->CheckInplace(i, 0))
                        InplaceTraversal(v, ancestor);
    }
}

}  // namespace dragon