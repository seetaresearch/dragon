#include "core/operator_schema.h"
#include "core/graph.h"
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

    //  topo-check for a graph
    for (const OperatorDef& op : meta_graph.op()) {
        //  check inputs
        for (auto& in : op.input())
            CHECK(known_tensors.count(in) || ws_->HasTensor(in))
                << "\nInput: " << in << " for op: " 
                << op.name() << " is unknown.";
        //  add outputs
        for (auto& out : op.output()) known_tensors.insert(out);
    }

    //  check for all objective targets
    Set<string> objective_targets;
    for (auto& target : meta_graph.target()) {
        CHECK(known_tensors.count(target) || ws_->HasTensor(target))
            << "\nTarget: " << target << " does not exist in computional graph.";
        objective_targets.insert(target);
    }

    //  check for all gradient targets
    for (auto& g_target : meta_graph.g_target()) {
        string cost = g_target.cost();
        string wrt = g_target.wrt();
        CHECK(known_tensors.count(cost) || ws_->HasTensor(cost))
            << "\nTarget: " << cost << "_grad does not exist in computional graph.";
        CHECK(known_tensors.count(wrt) || ws_->HasTensor(wrt))
            << "\nTarget: " << wrt << "_grad does not exist in computional graph.";
        CHECK_GT(objective_targets.count(cost), 0)
            << "\nTo solve d(" << cost << ")/d(" << wrt << "), "
            << "must set " << cost
            << "\nas a objective tensor to solve before derivating it.";
    }
}

void Graph::ForwardShareDyeing(string u, string ancestor) {
    if (renamed_.count(u)) return;
    renamed_[u] = ancestor;
    if (dag_[u].childs.size() == 1) {
        string op_type = dag_[dag_[u].childs[0]].op_type;
        auto* schema = OpSchemaRegistry::Schema(op_type);
        if (schema->AllowInplace())
            ForwardShareDyeing(dag_[u].childs[0], ancestor);
    }        
}

void Graph::ForwardPruneDyeing(string u, string leaf, vector<string> path) {
    if (visited_.count(u)) {
        if (visited_[u]) 
            for (auto& node : path) 
                visited_[node] = colored_[node] = true;
        return;
    }
    visited_[u] = false;
    for (int i = 0; i < dag_[u].childs.size(); i++) {
        string v = dag_[u].childs[i];
        vector<string> new_path(path);
        new_path.push_back(v);
        if (v == leaf) {
            for (auto& node : new_path) 
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
    //  build Graph
    for (int i = 0; i < meta_graph.op_size(); i++) {
        const OperatorDef& op = meta_graph.op(i);
        for (auto& v : op.output()) {
            vector<string> sp_u;
            if (!op.input_size()) sp_u.resize(op.output_size(), "");
            else sp_u.assign(op.input().begin(), op.input().end());
            for (auto& u : sp_u) {
                if (u == "ignore") continue;
                dag_[v].parents.push_back(u);
                dag_[u].childs.push_back(v);
                dag_[v].op_idx = i;
            }
            dag_[v].op_type = op.type();
        }
    }

    //  backward dyeing for all objective targets (e.g. loss)
    for (int i = 0; i < meta_graph.target_size(); i++) {
        targets_.insert(meta_graph.target(i));
        if (colored_[meta_graph.target(i)]) continue;
        BackwardPruneDyeing(meta_graph.target(i));
    }

    //  forward dyeing through connected path for all gradient targets
    for (int i = 0; i < meta_graph.g_target_size(); i++) {
        targets_.insert(meta_graph.g_target(i).wrt() + "_grad");
        string u = meta_graph.g_target(i).cost() + "_grad";
        string v = meta_graph.g_target(i).wrt() + "_grad";
        if (!meta_graph.g_target(i).external().empty())
            v = meta_graph.g_target(i).external();
        if (ws()->HasTensor(u)) u = ws()->GetTensor(u)->name();
        if (ws()->HasTensor(v)) v = ws()->GetTensor(v)->name();
        visited_.clear();
        ForwardPruneDyeing(u, v, vector<string>({ u }));
    }

    //  select all colored operators
    set<int> selected_op_indices;    //  note that we use set to keep topo-order
    for (auto it : colored_) {
        if (dag_[it.first].op_idx == -1) continue;
        selected_op_indices.insert(dag_[it.first].op_idx);
    }

    //  ensure that the update target will not be removed(color it)
    for (int i = 0; i < meta_graph.u_target_size(); i++) {
        UpdateTarget target = meta_graph.u_target(i);
        for (auto& tensor : target.tensor())
            colored_[tensor] = true;
    }

    //  remove the tensors that can not be produced(redundant)
    Set<string> outputs;
    //  check if having feeded tensors
    for (auto& tensor : ws()->GetTensors()) outputs.insert(tensor);
    map<int, OperatorDef> ops_final;  //  note that we use map to keep topo-order

    for (auto it : selected_op_indices) {
        OperatorDef op_def;
        op_def.CopyFrom(meta_graph.op(it));
        //  handle inputs
        for (int i = 0; i < meta_graph.op(it).input_size(); i++) {
            string input = meta_graph.op(it).input(i);
            if (!colored_[input] || !outputs.count(input))
                *op_def.mutable_input(i) = "ignore";
        }
        //  handle outputs
        for (int i = 0; i < meta_graph.op(it).output_size(); i++) {
            string output = meta_graph.op(it).output(i);
            if (!colored_[output]) *op_def.mutable_output(i) = "ignore";
            else outputs.insert(op_def.output(i));
        }
        ops_final[it].CopyFrom(op_def);
    }

    //  build the pruned graph
    GraphDef pruned_graph;
    pruned_graph.CopyFrom(meta_graph);
    pruned_graph.clear_op();
    for (auto it : ops_final) pruned_graph.add_op()->CopyFrom(it.second);
    return pruned_graph;
}

GraphDef Graph::Share(const GraphDef& optimized_graph) {
    renamed_.clear();

    //  forward dyeing to search available tensors that be shared
    for (int i = 0; i < optimized_graph.op_size(); i++) {
        const OperatorDef& op = optimized_graph.op(i);
        for (auto& u : op.input()) ForwardShareDyeing(u, u);
        for (auto& v : op.output()) ForwardShareDyeing(v, v);
    }

    GraphDef shared_graph;
    shared_graph.CopyFrom(optimized_graph);

    //  rename to create in-place
    for (int i = 0; i < optimized_graph.op_size(); i++) {
        const OperatorDef& op = optimized_graph.op(i);
        for (int j = 0; j < op.input_size(); j++) {
            if (renamed_.count(op.input(j))) {
                *shared_graph.mutable_op(i)->mutable_input(j) = renamed_[op.input(j)];
                ws()->CreateRename(op.input(j), renamed_[op.input(j)]);
            }
        }
        for (int j = 0; j < op.output_size(); j++) {
            if (renamed_.count(op.output(j))) {
                *shared_graph.mutable_op(i)->mutable_output(j) = renamed_[op.output(j)];
                ws()->CreateRename(op.output(j), renamed_[op.output(j)]);
            }
        }
    }
    return shared_graph;
}

GraphDef Graph::MakeUpdate(const GraphDef& meta_graph) {
    OperatorDef collective_op;
    collective_op.set_type("CollectiveUpdate");

    //  make update ops
    vector<OperatorDef> update_ops;
    for (int i = 0; i < meta_graph.u_target_size(); i++) {
        UpdateTarget target = meta_graph.u_target(i);
        vector<string> missing_tensors;
        for (auto& tensor : target.tensor()) {
            if (!ws()->HasTensor(tensor)) {
                LOG(INFO) << "Missing Tensor: " << tensor;
                missing_tensors.push_back(tensor);
            }
        }
        if (missing_tensors.size() == 0) {
            vector<Argument> args;
            for (auto& arg : target.arg()) args.push_back(arg);
            OperatorDef op_def = MakeOperatorDef(target.type(),
                                                 target.name(),
                          vector<string>({ target.tensor(1) }),  // dx
                          vector<string>({ target.tensor(0) })); // x
            collective_op.add_input(target.tensor(1));
            collective_op.add_output(target.tensor(1));
            op_def.mutable_arg()->CopyFrom(target.arg());
            update_ops.push_back(op_def);
        } else {
            LOG(INFO) << "Missing tensors. Skip update Tensor("
                      << target.tensor(0) << ")";
        }
    }

    //  make collective ops if necessary
    vector<OperatorDef> collective_ops;
    if (this->args_.count("parallel_mode")) {
        if (this->args_["parallel_mode"].s() == "MPI" ||
            this->args_["parallel_mode"].s() == "NCCL") {
            OperatorDef op_def;
            op_def.CopyFrom(collective_op);
            Argument collective_mode;
            collective_mode.set_name("mode");
            collective_mode.set_s(this->args_["parallel_mode"].s() + "_ALLREDUCE");
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
            /*
                See:  Accurate, Large Minibatch SGD: Training ImageNet in 1 Hour 
                Links: http://arxiv.org/abs/1706.02677
            */
            NOT_IMPLEMENTED;
        }
    }

    //  generate graph
    GraphDef update_graph;
    update_graph.CopyFrom(meta_graph);
    update_graph.clear_u_target();
    for (auto& op : collective_ops) update_graph.add_op()->CopyFrom(op);
    for (auto& op : update_ops) update_graph.add_op()->CopyFrom(op);
    return update_graph;
}

bool Graph::Create(const GraphDef& optimized_graph, Workspace* ws) {
    bool has_device_option = optimized_graph.has_device_option();
    bool has_debug_mode = optimized_graph.has_debug_mode();
    bool has_share_grads = optimized_graph.has_share_grads();
    for (const OperatorDef& plain_op_def : optimized_graph.op()) {
        OperatorDef op_def(plain_op_def);
        LOG(DEBUG) << "Create Operator " << plain_op_def.name() 
                   << ": " << plain_op_def.type();

        //  inherit device option if necessary
        if (!op_def.has_device_option() && has_device_option) 
            op_def.mutable_device_option()->CopyFrom(optimized_graph.device_option());

        //  inherit debug mode if necessary
        if (!op_def.has_debug_mode() && has_debug_mode)
            op_def.set_debug_mode(optimized_graph.debug_mode());

        //  inherit share_grads if necessary
        if (!op_def.has_share_grads() && has_share_grads)
            op_def.set_share_grads(optimized_graph.share_grads());

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

    //  check mirror stage
    for (int i = 0; i < ops_.size(); i++) {
        if (ops_[i]->type().find("Gradient") != string::npos) continue;
        bool mirror_stage = ops_[i]->GetSingleArg<bool>("mirror_stage", false);
        for (auto& u : optimized_graph.op(i).input()) {
            bool inplace_flag = false;
            for (auto& v : optimized_graph.op(i).output()) 
                if (u == v) inplace_flag = true;
            mirror_stage &= (!inplace_flag);
            if (!inplace_flag) multi_use_count[u]++;
        }
        if (mirror_stage) {
            //  TODO(PhyscalX):  we assume input(0)->output(0) as a in-place currently
            OperatorDef* op = fake_graph.mutable_op(i);
            if (rename_map.count(op->input(0))) 
                *op->mutable_input(0) = rename_map[op->input(0)];
            rename_map[op->output(0)] = op->input(0);
            *op->mutable_output(0) = op->input(0);
            ops_[i]->input(0).Corrupt();    //  mark as a flag
        }
    }

    //  sub-graph aware
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
   
    //  prepare resources
    for (auto& ops : ops_) ops->set_recompute_map(recompute_map);
    Tensor* head = ws->CreateTensor("/opt/mirror_stage/head");
    head->Reshape(vector<TIndex>(1, WORKSPACE_MAX_CORRUPTED_SIZE));
    Tensor* recompute_flag = ws->CreateTensor("/opt/mirror_stage/recompute_flag");
    recompute_flag->Reshape(vector<TIndex>(1, 1));
    recompute_flag->mutable_data<bool, CPUContext>()[0] = false;
    for (int i = 0; i < WORKSPACE_MAX_CORRUPTED_SIZE; i++) {
        string name = "/opt/mirror_stage/buffer_" + dragon_cast<string, int>(i);
        Tensor* buffer = ws->CreateTensor(name);
        head->mutable_data<string, CPUContext>()[i] = "";
    }
}

Graph::Graph(const GraphDef& meta_graph, Workspace* ws)
    : GraphBase(meta_graph, ws) {
    GraphDef optimized_graph;
    if (meta_graph.u_target_size() > 0) {
        //  check if existing any update requests
        //  note that graph with update ops is not a dag
        //  we handle them independently
        optimized_graph = MakeUpdate(meta_graph);
    } else {
        optimized_graph = Prune(meta_graph);
        optimized_graph = Share(optimized_graph);
    }

    //  store the final graph as a tensor for visualization
    Tensor* string_tensor = ws_->CreateTensor("GraphDef_" + optimized_graph.name());
    string_tensor->Reshape(vector<TIndex>(1, 1));
    string* data = string_tensor->mutable_data<string, CPUContext>();
    data[0] = optimized_graph.SerializeAsString();

    //  create
    Create(optimized_graph, ws);

    //  recomputing-aware
    RecomputingAware(optimized_graph, ws);
}

bool Graph::Run(const string& include, const string& exclude) {
    LOG(DEBUG) << "Run Graph: " << name();
    for (auto op : ops_) {
        if (!include.empty()) 
            if (op->type().find(include) == string::npos) continue;
        if (!exclude.empty())
            if (op->type().find(exclude) != string::npos) continue;
        op->SwitchToPhase(this->args_["phase"].s());
        LOG(DEBUG) << "$ Before Operator: " << op->name();
        op->Run();
        LOG(DEBUG) << "$ After Operator: " << op->name();
    }
    return true;
}

DEFINE_REGISTRY(GraphRegistry, GraphBase, const GraphDef&, Workspace*);

GraphBase* NewGraph(const GraphDef& meta_graph, Workspace* ws) {
    if (!meta_graph.has_graph_type()) return new Graph(meta_graph, ws);
    return GraphRegistry()->Create(meta_graph.graph_type(), meta_graph, ws);
}

}    // namespace dragon