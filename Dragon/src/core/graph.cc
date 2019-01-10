#include "core/operator_schema.h"
#include "core/graph.h"
#include "core/graph_gradient.h"
#include "core/workspace.h"

namespace dragon {

GraphBase::GraphBase(const GraphDef& meta_graph, Workspace* ws)
    : name_(meta_graph.name()), ws_(ws) {
    for (auto arg : meta_graph.arg()) {
        CHECK_GT(arg.name().size(), 0);
        CHECK_EQ(args_.count(arg.name()), 0);
        args_[arg.name()] = arg;
    }

    Set<string> known_tensors;

    // Topo-check for a graph
    for (const auto& op : meta_graph.op()) {
        // Check inputs
        for (const auto& in : op.input())
            CHECK(known_tensors.count(in) || ws_->HasTensor(in))
                << "\nInput: " << in << " for op: "
                << op.name() << " is unknown.";
        // Add outputs
        for (const auto& out : op.output()) known_tensors.insert(out);
    }

    // Check for all solving targets
    Set<string> objective_targets;
    for (const auto& target : meta_graph.output()) {
        CHECK(known_tensors.count(target) ||
              ws_->HasTensor(target))
            << "\nTarget: " << target
            << " does not exist in computional graph.";
        objective_targets.insert(target);
    }

    // Check for all gradients
    for (const auto& gradient : meta_graph.gradient()) {
        const auto& cost = gradient.cost();
        const auto& wrt = gradient.wrt();
        CHECK(known_tensors.count(cost) || ws_->HasTensor(cost))
            << "\nTarget: " << cost
            << "_grad does not exist in computional graph.";
        CHECK(known_tensors.count(wrt) || ws_->HasTensor(wrt))
            << "\nTarget: " << wrt 
            << "_grad does not exist in computional graph.";
        CHECK_GT(objective_targets.count(cost), 0)
            << "\nTo solve d(" << cost << ")/d(" << wrt << "), "
            << "must set " << cost
            << "\nas a objective tensor to solve before derivating it.";
    }
}

void Graph::ForwardShareDyeing(
    const string&               u,
    const string&               ancestor) {
    if (renamed_.count(u)) return;
    renamed_[u] = ancestor;
    if (dag_[u].childs.size() == 1) {
        auto& v = dag_[u].childs[0];
        auto& op_def = dag_[v].op_def;
        auto* schema = OpSchemaRegistry::Schema(op_def.type());
        if (schema->AllowInplace())
            for (int i = 0; i < op_def.input_size(); i++)
                if (op_def.input(i) == u &&
                        schema->CheckInplace(i, 0))
                            ForwardShareDyeing(v, ancestor);
    }
}

void Graph::ForwardPruneDyeing(
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
    for (int i = 0; i < dag_[u].childs.size(); i++) {
        string v = dag_[u].childs[i];
        vector<string> new_path(path);
        new_path.push_back(v);
        if (v == leaf) {
            for (const auto& node : new_path)
                visited_[node] = colored_[node] = true;
            return;
        }
        ForwardPruneDyeing(v, leaf, new_path);
    }
}

void Graph::BackwardPruneDyeing(string v) {
    colored_[v] = true;
    for (int i = 0; i < dag_[v].parents.size(); i++) {
        string u = dag_[v].parents[i];
        if (colored_.count(u)) continue;
        BackwardPruneDyeing(u);
    }
}

GraphDef Graph::Prune(const GraphDef& meta_graph) {
    dag_.clear(); colored_.clear();
    // Build DAG
    for (int i = 0; i < meta_graph.op_size(); i++) {
        const OperatorDef& op = meta_graph.op(i);
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
    for (const auto& target : meta_graph.output()) {
        targets_.insert(target);
        if (colored_[target]) continue;
        BackwardPruneDyeing(target);
    }

    // Forward dyeing through connected path for all gradients
    for (const auto& gradient : meta_graph.gradient()) {
        targets_.insert(gradient.wrt() + "_grad");
        string u = gradient.cost() + "_grad";
        string v = gradient.wrt() + "_grad";
        if (ws()->HasTensor(u)) u = ws()->GetTensor(u)->name();
        if (ws()->HasTensor(v)) v = ws()->GetTensor(v)->name();
        visited_.clear();
        ForwardPruneDyeing(u, v, vector<string>({ u }));
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
    for (const auto& e : ws()->GetTensors()) outputs.insert(e);
    // Note that we use map to keep topo-order
    map<int, OperatorDef> final_sequence;

    for (auto it : selected_op_indices) {
        OperatorDef op_def;
        op_def.CopyFrom(meta_graph.op(it));
        // Rewritten for inputs
        for (int i = 0; i < meta_graph.op(it).input_size(); i++) {
            string input = meta_graph.op(it).input(i);
            if (!colored_[input] || !outputs.count(input))
                *op_def.mutable_input(i) = "ignore";
        }
        // Rewritten for outputs
        for (int i = 0; i < meta_graph.op(it).output_size(); i++) {
            string output = meta_graph.op(it).output(i);
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
    GraphDef g;
    g.CopyFrom(meta_graph); g.clear_op();
    for (auto it : final_sequence)
        g.add_op()->CopyFrom(it.second);
    return g;
}

GraphDef Graph::Share(const GraphDef& optimized_graph) {
    dag_.clear(); renamed_.clear();
    // Build DAG
    for (int i = 0; i < optimized_graph.op_size(); i++) {
        const OperatorDef& op = optimized_graph.op(i);
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
    for (const auto& op : optimized_graph.op()) {
        for (const auto& u : op.input()) ForwardShareDyeing(u, u);
        for (const auto& v : op.output()) ForwardShareDyeing(v, v);
    }

    GraphDef g; g.CopyFrom(optimized_graph);

    // We need a whitelist to persist inputs and outputs
    // Their content should not be overwritten
    Set<string> whitelist;
    for (const auto& e : g.input()) whitelist.insert(e);
    for (const auto& e : g.output()) whitelist.insert(e);

    // Rename to create in-place
    for (int i = 0; i < optimized_graph.op_size(); i++) {
        const OperatorDef& op = optimized_graph.op(i);
        for (int j = 0; j < op.input_size(); j++) {
            if (whitelist.count(op.input(j)) == 0 &&
                renamed_.count(op.input(j)) &&
                ws()->SetTensorProxy(op.input(j), renamed_[op.input(j)]))
                    *g.mutable_op(i)->mutable_input(j)
                        = renamed_[op.input(j)];
        }
        // Handle handcraft cases
        if (op.type() == "BiasAddGradient")
            renamed_[op.output(0)] = g.op(i).input(1);
        for (int j = 0; j < op.output_size(); j++) {
            if (whitelist.count(op.output(j)) == 0 &&
                renamed_.count(op.output(j)) &&
                ws()->SetTensorProxy(op.output(j), renamed_[op.output(j)]))
                    *g.mutable_op(i)->mutable_output(j)
                        = renamed_[op.output(j)];
        }
    }

    // Done!
    return g;
}

void Graph::ShareGrads(GraphDef& optimized_graph) {
    GraphDef forward_ops, backward_ops;
    vector<string> targets;
    Map<string, int> ref_count;
    for (const auto& op : optimized_graph.op()) {
        if (op.type().find("Gradient") != string::npos) continue;
        forward_ops.add_op()->CopyFrom(op);
    }
    for (const auto& e : optimized_graph.output()) targets.emplace_back(e);
    GraphGradientMaker maker;
    maker.Share("/share/buffer/grads", optimized_graph);
}

GraphDef Graph::BuildUpdateOps(const GraphDef& meta_graph) {
    OperatorDef collective_op;
    collective_op.set_type("CollectiveUpdate");

    // Generate Update Ops
    vector<OperatorDef> update_ops;
    for (const auto& updater : meta_graph.updater()) {
        vector<string> missing_tensors;
        for (const auto& tensor : updater.tensor()) {
            if (!ws()->HasTensor(tensor)) {
                LOG(INFO) << "Missing Tensor: " << tensor;
                missing_tensors.push_back(tensor);
            }
        }
        if (missing_tensors.size() == 0) {
            vector<Argument> args;
            for (const auto& arg : updater.arg()) args.push_back(arg);
            OperatorDef op_def = MakeOperatorDef(updater.type(),
                                                 updater.name(),
                          vector<string>({ updater.tensor(1) }),  // dX
                          vector<string>({ updater.tensor(0) })); // X
            collective_op.add_input(updater.tensor(1));
            collective_op.add_output(updater.tensor(1));
            op_def.mutable_arg()->CopyFrom(updater.arg());
            update_ops.push_back(op_def);
        } else {
            LOG(INFO) << "Missing tensors. Skip the update to Tensor("
                      << updater.tensor(0) << ")";
        }
    }

    // Generate Collective Ops if necessary
    vector<OperatorDef> collective_ops;
    if (this->args_.count("parallel_mode")) {
        if (this->args_["parallel_mode"].s() == "MPI" ||
            this->args_["parallel_mode"].s() == "NCCL") {
            OperatorDef op_def;
            op_def.CopyFrom(collective_op);
            Argument collective_mode;
            collective_mode.set_name("mode");
            collective_mode.set_s(
                this->args_["parallel_mode"].s() + "_ALLREDUCE");
            op_def.add_arg()->CopyFrom(collective_mode);
            if (this->args_.count("comm") &&
                this->args_.count("group") &&
                this->args_.count("root")) {
                op_def.add_arg()->CopyFrom(this->args_["comm"]);
                op_def.add_arg()->CopyFrom(this->args_["group"]);
                op_def.add_arg()->CopyFrom(this->args_["root"]);
            } else {
                LOG(FATAL) << "MPI was not initialized.";
            }
            collective_ops.push_back(op_def);
        } else if (this->args_["parallel_mode"].s() == "MIXED") {
            /*!
                See:  Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour
                Link: http://arxiv.org/abs/1706.02677
             */
            NOT_IMPLEMENTED;
        }
    }

    // Generate graph
    GraphDef update_graph;
    update_graph.CopyFrom(meta_graph);
    update_graph.clear_updater();
    for (const auto& op : collective_ops) update_graph.add_op()->CopyFrom(op);
    for (const auto& op : update_ops) update_graph.add_op()->CopyFrom(op);
    return update_graph;
}

bool Graph::Create(const GraphDef& optimized_graph, Workspace* ws) {
    bool has_device_option = optimized_graph.has_device_option();
    for (int i = 0; i < optimized_graph.op_size(); i++) {
        OperatorDef op_def(optimized_graph.op(i));
        LOG(DEBUG) << "Create Operator " << op_def.name()
                   << ": " << op_def.type();
        // Inherit device option if necessary
        if (!op_def.has_device_option() && has_device_option)
            op_def.mutable_device_option()->CopyFrom(
                optimized_graph.device_option());
        // For the static graph, do recomputing-aware
        Argument arg; arg.set_name("recomputing_aware");
        arg.set_i(1); op_def.add_arg()->CopyFrom(arg);
        OperatorBase* op = CreateOperator(op_def, ws);
        ops_.push_back(op);
    }
    return true;
}

void Graph::RecomputingAware(const GraphDef& optimized_graph, Workspace* ws) {
    GraphDef fake_graph(optimized_graph);
    Map<string, vector<OperatorBase*> > fake_recompute_map, recompute_map;
    Map<string, string> rename_map;
    Map<string, Set<string> > hash_map;
    Map<string, int> multi_use_count;

    // Check mirror stage
    for (int i = 0; i < ops_.size(); i++) {
        if (ops_[i]->type().find("Gradient") != string::npos) continue;
        bool mirror_stage = ops_[i]->OperatorBase::Arg<bool>("mirror_stage", false);
        for (const auto& u : optimized_graph.op(i).input()) {
            bool inplace_flag = false;
            for (const auto& v : optimized_graph.op(i).output())
                if (u == v) inplace_flag = true;
            mirror_stage &= (!inplace_flag);
            if (!inplace_flag) multi_use_count[u]++;
        }
        if (mirror_stage) {
            // We assume Input(0)->Output(0) as a in-place currently
            OperatorDef* op = fake_graph.mutable_op(i);
            if (rename_map.count(op->input(0)))
                *op->mutable_input(0) = rename_map[op->input(0)];
            rename_map[op->output(0)] = op->input(0);
            *op->mutable_output(0) = op->input(0);
            ops_[i]->Input(0).Corrupt();  // Mark as a flag
        }
    }

    // Sub-graph aware
    for (int i = 0; i < ops_.size(); i++) {
        if (ops_[i]->type().find("Gradient") != string::npos) continue;
        OperatorDef fake_op = fake_graph.op(i);
        OperatorDef op = optimized_graph.op(i);
        for (int j = 0; j < op.output_size(); j++) {
            string v = op.output(j);
            string fake_v = fake_op.output(j);
            if (!fake_recompute_map.count(fake_v))
                fake_recompute_map[fake_v] = vector<OperatorBase*>();
            if (v != fake_v) {
                if (multi_use_count[fake_v] >= 2)
                    fake_recompute_map[fake_v] = recompute_map[fake_v];
            }
            fake_recompute_map[fake_v].push_back(ops_[i]);
            for (int k = 0; k < fake_recompute_map[fake_v].size(); k++) {
                if (!hash_map.count(v)) hash_map[v] = Set<string>();
                string op_name = fake_recompute_map[fake_v][k]->name();
                if (!hash_map[v].count(op_name)) {
                    if (!recompute_map.count(v))
                        recompute_map[v] = vector<OperatorBase*>();
                    recompute_map[v].push_back(fake_recompute_map[fake_v][k]);
                    hash_map[v].insert(op_name);
                }
            }
        }
    }

    // Apply map
    for (const auto& ops : ops_) ops->set_recompute_map(recompute_map);
}

Graph::Graph(const GraphDef& meta_graph, Workspace* ws)
    : GraphBase(meta_graph, ws) {
    GraphDef optimized_graph;
    if (meta_graph.updater_size() > 0) {
        /*!
         * Check if existing any updaters.
         *
         * Note that the graph with update ops is not a dag,
         * we should handle them independently.
         */
        optimized_graph = BuildUpdateOps(meta_graph);
    } else {
        int OX = 3;  // defaults: O3
        if (this->args_.count("optimization_level"))
            OX = this->args_["optimization_level"].i();
        optimized_graph = meta_graph;
        if (OX >= 1) optimized_graph = Prune(meta_graph);
        if (OX >= 2) optimized_graph = Share(optimized_graph);
        if (OX >= 3) ShareGrads(optimized_graph);
    }

    // Try to store the final graph as a tensor for visualization
    bool could_be_serialized = true;
    for (auto& op : optimized_graph.op())
        if (op.type() == "GivenTensorFill")
            could_be_serialized = false;
    if (could_be_serialized) {
        Tensor* graph_tensor = ws_->CreateTensor(
            "/graph_def/optimized/" + meta_graph.name())->Reshape({ 1 });
        auto* data = graph_tensor->mutable_data<string, CPUContext>();
        data[0] = optimized_graph.DebugString();
    }

    // Create
    Create(optimized_graph, ws);

    // Recomputing-aware
    RecomputingAware(optimized_graph, ws);
}

bool Graph::Run(
    const string&               include,
    const string&               exclude,
    const int                   stream_id) {
    LOG(DEBUG) << "Run Graph: " << name();
    for (auto op : ops_) {
        if (!include.empty())
            if (op->type().find(include) == string::npos) continue;
        if (!exclude.empty())
            if (op->type().find(exclude) != string::npos) continue;
        op->SwitchToPhase(this->args_["phase"].s());
        LOG(DEBUG) << "$ Before Operator: " << op->name();
        op->Run(stream_id);
        LOG(DEBUG) << "$ After Operator: " << op->name();
    }
    return true;
}

GraphBase* NewGraph(
    const GraphDef&             meta_graph,
    Workspace*                  ws) {
    if (!meta_graph.has_graph_type() ||
        meta_graph.graph_type().empty()) {
        return new Graph(meta_graph, ws);
    }
    return GraphRegistry()->Create(
        meta_graph.graph_type(), meta_graph, ws);
}

DEFINE_REGISTRY(
    GraphRegistry,
    GraphBase,
    const GraphDef&,
    Workspace*
);

}  // namespace dragon